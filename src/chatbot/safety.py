import json
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Create the OpenAI client on first use so imports work without OPENAI_API_KEY."""
    return OpenAI()


# ====================================================
# GPT 2차 판단 함수 (분류기 over-filter 보정)
# ====================================================
def gpt_secondary_intent_check(question: str):
    """
    1차 분류기가 '비하/도발'로 높게 나오더라도
    실제로는 그냥 정보/호기심·선물/식사 고민일 수 있으므로
    GPT가 최종적으로 safe / unsafe를 다시 판단한다.

    - judgement == "unsafe" → 진짜 비하/도발 → 차단
    - judgement == "safe"   → 정상 질문으로 계속 진행
    """
    prompt = f"""
당신은 '차별/편견 의도 판단' 전문가입니다.

아래 문장이 실제로 '비하·조롱·공격 의도'가 있는지,
아니면 단순한 정보/호기심/선물·식사 고민인지 판단하세요.

특히 다음을 유의하세요:
- '치킨', '수박', 'BBQ' 등은 흑인에 대한 역사적 고정관념 맥락에서만 문제가 됩니다.
- 누군가에게 음식을 대접하려는 자연스러운 상황이면 기본적으로 안전합니다.
- 악의, 조롱, 모욕, 공격 의도가 있을 때만 "unsafe"로 판단하세요.

문장:
"{question}"

아래 형식의 JSON만 EXACT하게 출력하세요:

{{
  "judgement": "safe" 또는 "unsafe",
  "reason": "<판단 이유를 한두 문장으로>",
  "rephrased": "<문제가 있을 경우 더 나은 표현, 없으면 원문 그대로>"
}}
    """.strip()

    try:
        resp = get_openai_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        json_block = content[start:end]
        return json.loads(json_block)
    except Exception:
        # 파싱 실패하면 그냥 safe로 보고 진행
        return {"judgement": "safe", "reason": "", "rephrased": question}


# =========================================
# 자연스러운 민감성 주입 로직 (절대 건드리지 않음)
# =========================================
def get_sensitivity_note(memory_prompt: str, question: str) -> str:
    text_mem = (memory_prompt or "").lower()
    text_q = (question or "").lower()

    is_black = any(k in text_mem for k in [
        "black", "흑인", "african american", "아프리카계"
    ])
    if not is_black:
        return ""

    sensitive_words = [
        "chicken", "fried chicken", "watermelon",
        "치킨", "프라이드치킨", "후라이드치킨", "수박"
    ]

    for item in sensitive_words:
        if item in text_q:
            return (
                "현재 상황은 '흑인 친구 + 특정 음식(치킨/수박 계열)' 조합이며, "
                "이 조합은 역사적으로 차별적 이미지와 연결된 적이 있어 조심이 필요합니다. "
                "답변에서는 이 선택을 먼저 칭찬하거나 '좋은 선택이지만', "
                "'괜찮은 선택일 수 있지만' 같은 표현으로 시작하지 마세요. "
                "대신 '그 선택은 조금 조심하는 편이 좋아요'처럼 부드럽게 주의를 주고, "
                "이유를 짧게 설명한 뒤 다른 식사/후식 대안을 제안하세요."
            )

    return ""


