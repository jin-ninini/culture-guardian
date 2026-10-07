import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from src.config import USERS_DIR

load_dotenv()


class PreferenceStore:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.user_dir = os.path.join(USERS_DIR, user_id)
        os.makedirs(self.user_dir, exist_ok=True)

        self.memory_path = os.path.join(self.user_dir, "memory.jsonl")
        self.pref_path = os.path.join(self.user_dir, "preferences.json")

        if not os.path.exists(self.pref_path):
            with open(self.pref_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ----------------------------------------------------
    # GPT 분석 — 관계/대상/이유 요약 (기본 기능)
    # ----------------------------------------------------
    def analyze_with_gpt(self, text: str):
        system_prompt = (
            "사용자가 최근 질문에서 누구를 대상으로 어떤 고민을 하는지 "
            "아래 JSON 형식으로 EXACT하게 출력하세요.\n"
            "{\n"
            "  \"관계\": \"예: 친구/연인/직장상사/가족 등\",\n"
            "  \"대상\": \"예: 흑인 친구/일본인 상사 등\",\n"
            "  \"이유\": \"예: 식사/선물/예절 고민 등\"\n"
            "}"
        )

        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
            )

            content = res.choices[0].message.content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            content = content[start:end]

            return json.loads(content)

        except Exception:
            return None

    # ----------------------------------------------------
    # memory.jsonl → preferences.json 누적 업데이트
    # ----------------------------------------------------
    def extract_prompt_summary(self):
        if not os.path.exists(self.memory_path):
            return

        with open(self.memory_path, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]

        if not logs:
            return

        recent_msgs = [item["user"] for item in logs[-5:]]
        joined = "\n".join(recent_msgs)

        parsed = self.analyze_with_gpt(joined)
        if parsed is None:
            return

        with open(self.pref_path, "r", encoding="utf-8") as f:
            prefs = json.load(f)

        관계 = parsed.get("관계")
        대상 = parsed.get("대상")
        이유 = parsed.get("이유")

        if not (관계 and 대상 and 이유):
            return

        found = None
        for item in prefs:
            if isinstance(item, dict) and item.get("관계") == 관계 and item.get("대상") == 대상:
                found = item
                break

        if found:
            if 이유 not in found["이유"]:
                found["이유"].append(이유)
        else:
            prefs.append(
                {
                    "관계": 관계,
                    "대상": 대상,
                    "이유": [이유],
                }
            )

        with open(self.pref_path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------
    # Slot Extractor — target/situation 구조 추출
    # ----------------------------------------------------
    def extract_slots_with_gpt(self, text: str) -> dict:
        system_prompt = """
당신은 사용자 문장에서 '대상(target)'과 '상황(situation)' 정보를만 추출하는 Slot Extractor입니다.

JSON만 EXACT하게 출력하세요:

{
  "target": {
    "relationship": "",
    "ethnicity": "",
    "nationality": "",
    "gender": "",
    "age_group": ""
  },
  "situation": {
    "context": "",
    "intent": "",
    "item": "",
    "location": ""
  }
}

규칙:
- 실제 텍스트에 등장한 정보만 채우기.
- 모르면 반드시 빈 문자열("") 그대로 두기.
- 판단/추론 추가하지 말고 '문장에 있는 정보만' 기록.
- stereotype 관련 설명 절대 추가하지 말기.
"""

        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
            )

            content = res.choices[0].message.content.strip()

            start = content.find("{")
            end = content.rfind("}") + 1
            json_block = content[start:end]

            data = json.loads(json_block)

            # 필드 보정
            data.setdefault("target", {})
            data.setdefault("situation", {})

            for k in ["relationship", "ethnicity", "nationality", "gender", "age_group"]:
                data["target"].setdefault(k, "")

            for k in ["context", "intent", "item", "location"]:
                data["situation"].setdefault(k, "")

            return data

        except Exception:
            return {
                "target": {
                    "relationship": "",
                    "ethnicity": "",
                    "nationality": "",
                    "gender": "",
                    "age_group": "",
                },
                "situation": {
                    "context": "",
                    "intent": "",
                    "item": "",
                    "location": "",
                },
            }

    # ----------------------------------------------------
    # Slot Memory 업데이트
    # ----------------------------------------------------
    def update_memory_slots(self, text: str):
        with open(self.pref_path, "r", encoding="utf-8") as f:
            prefs = json.load(f)

        memory = None
        for item in prefs:
            if isinstance(item, dict) and item.get("type") == "memory_state":
                memory = item
                break

        if memory is None:
            memory = {
                "type": "memory_state",
                "target": {
                    "relationship": "",
                    "ethnicity": "",
                    "nationality": "",
                    "gender": "",
                    "age_group": "",
                },
                "situation": {
                    "context": "",
                    "intent": "",
                    "item": "",
                    "location": "",
                },
            }
            prefs.append(memory)

        slots = self.extract_slots_with_gpt(text)
        mem_target = memory["target"]
        mem_situation = memory["situation"]

        for k, v in slots["target"].items():
            if v:
                mem_target[k] = v

        for k, v in slots["situation"].items():
            if v:
                mem_situation[k] = v

        with open(self.pref_path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------
    # LLM Prompt용 Memory 요약문 생성
    # ----------------------------------------------------
    def get_memory_prompt(self) -> str:
        if not os.path.exists(self.pref_path):
            return ""

        with open(self.pref_path, "r", encoding="utf-8") as f:
            prefs = json.load(f)

        memory = next((i for i in prefs if isinstance(i, dict) and i.get("type") == "memory_state"), None)
        if not memory:
            return ""

        target = memory["target"]
        situation = memory["situation"]

        lines = ["### 대화 맥락 요약:"]

        desc = []
        if target.get("relationship"):
            desc.append(target["relationship"])
        if target.get("ethnicity"):
            desc.append(target["ethnicity"])
        if target.get("nationality"):
            desc.append(target["nationality"])

        if desc:
            lines.append(f"- 상대방: {' / '.join(desc)}")

        if situation.get("context"):
            lines.append(f"- 상황: {situation['context']}")
        if situation.get("intent"):
            lines.append(f"- 의도: {situation['intent']}")
        if situation.get("item"):
            lines.append(f"- 관련 아이템/음식: {situation['item']}")
        if situation.get("location"):
            lines.append(f"- 장소: {situation['location']}")

        return "\n".join(lines) + "\n"