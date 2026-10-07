import torch.nn as nn
from transformers import AutoModel, AutoConfig


class IntentClassifier(nn.Module):
    def __init__(self, model_name_or_dir: str, num_labels: int = 3, dropout: float = 0.1):
        """
        model_name_or_dir: 허깅페이스 모델 이름 또는 save_pretrained 경로
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_dir)
        self.config.num_labels = num_labels

        # KoBERT 호환 (trust_remote_code)
        self.backbone = AutoModel.from_pretrained(
            model_name_or_dir,
            config=self.config,
            trust_remote_code=True
        )

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled = (
            outputs.pooler_output
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None
            else outputs.last_hidden_state[:, 0]
        )
        x = self.dropout(pooled)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
