import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import torch.nn.functional as F

from src.intent_classifier.dataset import IntentDataset
from src.intent_classifier.model import IntentClassifier
from src.config import experiment_dir
from src.intent_classifier.metrics import compute_classification_metrics, purity_score
from src.utils import get_logger, set_seed


class IntentTrainer:
    def __init__(self, config: dict):
        self.config = config
        set_seed(config.get("seed", 42))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = config["model_name"]
        self.is_kobert = "kobert" in self.model_name.lower()

        # KoBERT 모델은 trust_remote_code 필요
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            trust_remote_code=True
        )

        max_len = config["max_length"]
        self.train_dataset = IntentDataset(config["train_path"], self.tokenizer, max_len)
        self.valid_dataset = IntentDataset(config["valid_path"], self.tokenizer, max_len)

        self.model = IntentClassifier(
            self.model_name,
            num_labels=int(config["num_labels"]),
            dropout=float(config.get("dropout", 0.1)),
        ).to(self.device)

        # 하이퍼파라미터
        self.batch_size = int(config.get("batch_size", 32))
        self.lr = float(config.get("learning_rate", 3e-5))
        self.epochs = int(config.get("epochs", 3))
        self.weight_decay = float(config.get("weight_decay", 0.01))
        self.warmup_ratio = float(config.get("warmup_ratio", 0.1))

        # 경로 & 로깅
        self.save_dir = experiment_dir(config)
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.save_dir, "training.log"))
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "runs"))

        # dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

        # optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(params, lr=self.lr)

        # scheduler
        total_steps = len(self.train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # ===== R-Drop / Label Smoothing / EMA 설정 =====
        self.label_smoothing = float(self.config.get("label_smoothing", 0.1))
        self.rdrop_alpha = float(self.config.get("rdrop_alpha", 5.0))

        self.use_ema = bool(self.config.get("use_ema", True))
        self.ema_decay = float(self.config.get("ema_decay", 0.999))
        self.ema_params = None
        if self.use_ema:
            self.ema_params = {
                name: p.detach().clone()
                for name, p in self.model.named_parameters()
                if p.requires_grad
            }


    # ======================================================
    # 학습 루프
    # ======================================================
    def train(self):
        best_f1 = 0.0
        for epoch in range(1, self.epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.epochs}")
            train_loss = self._train_one_epoch(epoch)
            val_metrics = self._evaluate_loader(self.valid_loader, "Validation")

            test_metrics = None
            test_path = self.config.get("test_path")
            if test_path and os.path.exists(test_path):
                test_dataset = IntentDataset(test_path, self.tokenizer, self.config["max_length"])
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                test_metrics = self._evaluate_loader(test_loader, "Test")
            else:
                test_metrics = {"accuracy": 0, "f1": 0, "precision": 0, "recall": 0, "mcc": 0, "purity": 0}

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("F1/Validation", val_metrics["f1"], epoch)
            self.writer.add_scalar("Accuracy/Validation", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("F1/Test", test_metrics["f1"], epoch)
            self.writer.add_scalar("Accuracy/Test", test_metrics["accuracy"], epoch)

            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Metrics: {val_metrics}")
            self.logger.info(f"Test Metrics: {test_metrics}")

            # best 저장 (.pt + HF 구조)
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                self._save_best_model()
                self.logger.info(f"New best model saved with F1={best_f1:.4f}")

        self.writer.close()

# 내가 추가한 부분 --
    def _label_smoothing_loss(self, logits, labels):
        """라벨 스무딩이 들어간 CE loss"""
        if self.label_smoothing <= 0.0:
            return F.cross_entropy(logits, labels)

        eps = self.label_smoothing
        n_class = logits.size(-1)

        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(eps / (n_class - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - eps)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

    def _update_ema(self):
        """EMA 파라미터 갱신"""
        if not self.use_ema or self.ema_params is None:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.ema_params:
                    self.ema_params[name].mul_(self.ema_decay).add_(
                        param.data, alpha=1.0 - self.ema_decay
                    )

    def _swap_to_ema_weights(self):
        """평가할 때 EMA weight로 교체하고, 원래 weight는 백업"""
        if not self.use_ema or self.ema_params is None:
            return None  # backup 없음
        backup = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if name in self.ema_params
        }
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.ema_params:
                    param.copy_(self.ema_params[name])
        return backup

    def _restore_from_backup(self, backup):
        """EMA 평가 후 원래 weight 복구"""
        if backup is None:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in backup:
                    param.copy_(backup[name])

    # ======================================================
    # 한 epoch 학습
    # ======================================================
    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader, 1):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch["labels"]

            # KoBERT일 때 token_type_ids 처리
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            if self.is_kobert and "token_type_ids" in inputs:
                inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"]).to(self.device)

            # ===== R-Drop: 같은 입력으로 두 번 forward =====
            outputs1 = self.model(**inputs)
            outputs2 = self.model(**inputs)

            logits1 = outputs1["logits"]
            logits2 = outputs2["logits"]

            # 1) Label smoothing이 들어간 CE loss
            ce_loss1 = self._label_smoothing_loss(logits1, labels)
            ce_loss2 = self._label_smoothing_loss(logits2, labels)
            ce_loss = 0.5 * (ce_loss1 + ce_loss2)

            # 2) 양방향 KL loss (R-Drop)
            kl_loss = F.kl_div(
                F.log_softmax(logits1, dim=-1),
                F.softmax(logits2, dim=-1),
                reduction="batchmean",
            ) + F.kl_div(
                F.log_softmax(logits2, dim=-1),
                F.softmax(logits1, dim=-1),
                reduction="batchmean",
            )

            loss = ce_loss + self.rdrop_alpha * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

            # EMA 파라미터 업데이트
            self._update_ema()

            total_loss += loss.item()

            if step % self.config.get("log_interval", 50) == 0:
                self.logger.info(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

        return total_loss / len(self.train_loader)


    # ======================================================
    # 평가 공통 함수
    # ======================================================
    def _evaluate_loader(self, dataloader, split_name="Eval"):
        # EMA weight로 교체 (백업 반환)
        backup = self._swap_to_ema_weights()

        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                labels = batch["labels"].to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                inputs = {k: v for k, v in batch.items() if k != "labels"}

                if self.is_kobert and "token_type_ids" in inputs:
                    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"]).to(self.device)

                outputs = self.model(**inputs)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        # 평가 후 원래 weight 복원
        self._restore_from_backup(backup)

        metrics = compute_classification_metrics(all_labels, all_preds)
        metrics["purity"] = purity_score(all_labels, all_probs)
        self.logger.info(f"{split_name} Metrics: {metrics}")
        return metrics


    # ======================================================
    # 모델 저장 (HF + .pt)
    # ======================================================
    def _save_best_model(self):
        """평가/서빙용 best_model.pt + 토크나이저 저장"""

        # KoBERT 전용 토크나이저 저장 (별도 분기)
        try:
            self.tokenizer.save_pretrained(self.save_dir)
        except TypeError:
            # KoBERT의 경우 filename_prefix 인자를 지원하지 않음
            vocab_path = os.path.join(self.save_dir, "vocab.txt")
            if hasattr(self.tokenizer, "vocab"):
                with open(vocab_path, "w", encoding="utf-8") as f:
                    for token, idx in self.tokenizer.vocab.items():
                        f.write(f"{token}\n")
            # tokenizer_config.json도 함께 저장
            import json
            tok_cfg_path = os.path.join(self.save_dir, "tokenizer_config.json")
            with open(tok_cfg_path, "w", encoding="utf-8") as f:
                json.dump({"model_type": "kobert"}, f, ensure_ascii=False, indent=2)

            self.logger.info("KoBERT tokenizer saved manually.")

        pt_path = os.path.join(self.save_dir, "best_model.pt")
        # 검증 점수를 낸 EMA weight로 저장 (EMA 미사용 시 현재 weight)
        backup = self._swap_to_ema_weights()
        torch.save(self.model.state_dict(), pt_path)
        self._restore_from_backup(backup)
        self.logger.info(f"Saved best_model.pt and tokenizer to {self.save_dir}")


