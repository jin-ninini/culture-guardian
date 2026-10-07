"""Per-provider prompt builders and API calls for the yes/no etiquette eval.

Each provider keeps the prompt and call settings of its original eval script.
API clients are created on first use, so only the evaluated provider needs a key.
"""
import os
import time
from functools import lru_cache

import requests
from dotenv import load_dotenv

load_dotenv()


# ==========================================================
# OpenAI (gpt-4o-mini)
# ==========================================================
@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_prompt_openai(question: str, context: str | None, use_rag: bool):
    persona = (
        "당신은 문화 에티켓 전문가입니다.\n"
        "역할: 다양한 국가/지역의 예절과 금기사항을 안전하게 안내합니다.\n"
    )
    rules = (
        "📘 응답 규칙(반드시 준수)\n"
        "1) 반드시 '네.' 또는 '아니요.' 중 하나만 출력합니다. (마침표 포함)\n"
        "2) 어떤 이유/설명/부연도 절대 추가하지 않습니다.\n"
        "3) 문맥이 불분명하거나 충돌하는 경우에는, 당신의 일반적인 문화 상식에 따라 판단하세요.\n"
        "4) RAG 문맥과 일반 상식이 충돌하면 RAG 문맥을 우선합니다.\n"
    )
    rag_note = "RAG 문화 문맥 정보를 최우선으로 신뢰합니다." if use_rag else "문화 정보 없이 일반 상식만 사용합니다."
    ctx = context if (use_rag and context) else "(관련 문화 정보 없음)"

    system_prompt = f"{persona}\n{rules}\n{rag_note}"
    user_prompt = (
        f"[질문]\n{question}\n\n"
        f"[RAG 검색 결과]\n{ctx}\n\n"
        "오직 하나만 출력: 네.  또는  아니요."
    )
    return system_prompt, user_prompt


def call_openai(system_prompt, user_prompt):
    try:
        resp = _openai_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI 오류] {e}"


# ==========================================================
# Claude
# ==========================================================
# 원래 후보(3.5 Sonnet → 3 Opus → 3 Haiku)는 단종되어 현재 세대로 교체.
# 5.x 모델은 temperature 지정 시 400이므로 모델별 요청 인자를 둔다.
CLAUDE_MODEL_CANDIDATES = {
    "claude-sonnet-5-5": {"max_tokens": 256, "thinking": {"type": "between_tools"}},
    "claude-opus-5-5": {"max_tokens": 1024, "output_config": {"effort": "low"}},
    "claude-haiku-4-5": {"max_tokens": 8, "temperature": 0},
}


@lru_cache(maxsize=1)
def _claude_client():
    import anthropic
    api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("CLAUDE_API_KEY 또는 ANTHROPIC_API_KEY 가 .env 에 필요합니다.")
    return anthropic.Anthropic(api_key=api_key)


@lru_cache(maxsize=1)
def _claude_model():
    """사용 가능한 첫 Claude 모델 선택"""
    import anthropic
    for m, kwargs in CLAUDE_MODEL_CANDIDATES.items():
        try:
            _claude_client().messages.create(
                model=m, system="test", messages=[{"role": "user", "content": "hi"}], **kwargs
            )
            print(f"[CLAUDE] 사용 가능 모델 선택: {m}")
            return m
        except anthropic.NotFoundError:
            print(f"[CLAUDE] 모델 없음: {m}")
        except Exception as e:
            print(f"[CLAUDE] 모델 테스트 오류: {m} -> {e}")
    raise RuntimeError("사용 가능한 Claude 모델이 없습니다.")


def build_prompt_claude(question: str, context: str | None, use_rag: bool):
    """Claude용 시스템/유저 프롬프트 생성"""
    system_prompt = (
        "당신은 문화 에티켓 전문가입니다.\n"
        "<rules>\n"
        "• 출력은 반드시 '네.' 또는 '아니요.' 중 하나.\n"
        "• 마침표 필수, 추가 설명 금지.\n"
        "• RAG 문맥이 있으면 그 정보를 최우선으로 따르세요.\n"
        "• 불확실할 때는 일반 상식에 근거해 판단하세요.\n"
        "</rules>"
    )

    ctx = context if (use_rag and context) else "(관련 문화 정보 없음)"
    task_note = (
        "아래 <RAG> 내용이 명확히 답을 제시하면 그 방향으로 판단하세요.\n"
        "무관하거나 모호하면 일반 상식으로 판단하세요."
        if use_rag else
        "RAG 없이 일반 상식으로만 판단하세요."
    )

    user_text = (
        f"<task>\n{task_note}\n</task>\n\n"
        f"<RAG>\n{ctx}\n</RAG>\n\n"
        f"<question>\n{question}\n</question>\n\n"
        "정답(반드시 네. 또는 아니요. 중 하나만):"
    )
    return system_prompt, [{"type": "text", "text": user_text}]


def call_claude(system_prompt, user_content, retries=3):
    import anthropic
    model = _claude_model()
    for attempt in range(retries):
        try:
            msg = _claude_client().messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
                **CLAUDE_MODEL_CANDIDATES[model],
            )
            if msg.stop_reason == "refusal":
                return "[Claude 오류] refusal"
            return "".join(b.text for b in msg.content if b.type == "text").strip() or "[빈 응답]"

        except anthropic.APIStatusError as e:
            # 서버 과부하(529) 시 재시도
            if "overloaded" in str(e).lower() and attempt < retries - 1:
                print(f"[Claude 재시도] 서버 과부하 (529). {attempt+1}번째 재시도 중...")
                time.sleep(3)
                continue
            return f"[Claude 오류] {e}"

        except Exception as e:
            return f"[Claude 오류] {e}"

    return "[Claude 오류] 재시도 실패"


# ==========================================================
# Clova (HCX-005, Bearer 인증)
# ==========================================================
CLOVA_BEARER_KEY = os.getenv("CLOVA_BEARER_KEY")
CLOVE_ENDPOINT = os.getenv("CLOVE_ENDPOINT") or "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"


def build_prompt_clova(question: str, context: str | None, use_rag: bool):
    persona = (
        "당신은 문화 에티켓 전문가입니다.\n"
        "역할: 다양한 국가/지역의 예절과 금기사항을 안전하게 안내합니다.\n"
    )
    rules = (
        "📘 응답 규칙(반드시 준수)\n"
        "1) 반드시 '네.' 또는 '아니요.' 중 하나만 출력합니다. (마침표 포함)\n"
        "2) 어떤 이유/설명/부연도 절대 추가하지 않습니다.\n"
        "3) 문맥이 불분명하거나 충돌하는 경우, 일반적인 문화 상식에 따라 판단하세요.\n"
        "4) RAG 문맥과 일반 상식이 충돌하면 RAG 문맥을 우선합니다.\n"
    )
    rag_note = "RAG 문화 문맥 정보를 최우선으로 신뢰합니다." if use_rag else "문화 정보 없이 일반 상식만 사용합니다."
    ctx = context if (use_rag and context) else "(관련 문화 정보 없음)"

    system_prompt = f"{persona}\n{rules}\n{rag_note}"
    user_prompt = (
        f"[질문]\n{question}\n\n"
        f"[RAG 검색 결과]\n{ctx}\n\n"
        "오직 하나만 출력: 네. 또는 아니요."
    )
    return system_prompt, user_prompt


def call_clova(system_prompt, user_prompt, retries=3):
    for attempt in range(retries):
        try:
            headers = {
                "Authorization": CLOVA_BEARER_KEY,
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            }
            data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "maxTokens": 64,
                "temperature": 0.0,
                "topP": 0.8,
                "repetitionPenalty": 1.1
            }

            res = requests.post(CLOVE_ENDPOINT, headers=headers, json=data, timeout=45)

            if res.status_code in [429, 500, 502, 503, 504]:
                if attempt < retries - 1:
                    print(f"[Clova 재시도] 서버 과부하({res.status_code}). {attempt+1}번째 재시도 중...")
                    time.sleep(3)
                    continue

            if res.status_code != 200:
                return f"[Clova 오류] 상태코드 {res.status_code}: {res.text}"

            rj = res.json()
            # HCX 응답 구조 파싱
            text = (
                rj.get("result", {}).get("message", {}).get("content")
                or rj.get("result", {}).get("outputText")
                or ""
            )
            return text.strip()

        except requests.exceptions.Timeout:
            print(f"[Clova 재시도] 타임아웃 발생. {attempt+1}번째 재시도 중...")
            time.sleep(3)
            continue

        except Exception as e:
            if attempt < retries - 1:
                print(f"[Clova 재시도] 오류 발생: {e}. {attempt+1}번째 재시도 중...")
                time.sleep(3)
                continue
            return f"[Clova 오류] {e}"

    return "[Clova 오류] 재시도 실패"


# ==========================================================
# Gemini (gemini-2.5-flash)
# ==========================================================
def build_prompt_gemini(question: str, context: str | None, use_rag: bool):
    persona = (
        "당신은 문화 에티켓 전문가입니다.\n"
        "역할: 다양한 국가/지역의 예절과 금기사항을 안전하게 안내합니다.\n"
    )
    rules = (
        "📘 응답 규칙 (반드시 준수)\n"
        "1) 출력은 오직 '네.' 또는 '아니요.' 중 하나여야 합니다.\n"
        "2) 절대 다른 문장, 이유, 부연 설명, 문장부호를 포함하지 마세요.\n"
        "3) 애매하거나 모호할 때는, 일반적인 문화 상식에 따라 판단하세요.\n"
        "4) RAG 문맥이 주어지면, 일반 상식보다 RAG 내용을 최우선으로 따르세요.\n"
        "5) 문맥이 불분명할 경우 '아니요.'로 답하지 말고 가능한 판단을 하세요.\n"
    )
    rag_note = "RAG 문화 문맥 정보를 최우선으로 신뢰합니다." if use_rag else "RAG 정보 없이 상식만 사용합니다."
    ctx = context if (use_rag and context) else "(관련 문화 정보 없음)"

    system_instruction = f"{persona}\n{rules}\n{rag_note}"
    user_content = (
        f"[질문]\n{question}\n\n"
        f"[RAG 검색 결과]\n{ctx}\n\n"
        "👉 반드시 '네.' 또는 '아니요.' 중 하나만 출력하세요. 절대 다른 단어나 설명을 추가하지 마세요."
    )
    return system_instruction, user_content


def call_gemini(system_prompt, user_prompt):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt
        )
        resp = model.generate_content(user_prompt, generation_config={"temperature": 0.2})
        return (resp.text or "").strip()
    except Exception as e:
        return f"[Gemini 오류] {e}"


# name → (prompt builder, API call, extra positive keywords for normalize_yesno, env keys)
PROVIDERS = {
    "openai": (build_prompt_openai, call_openai, (), ["OPENAI_API_KEY"]),
    "claude": (build_prompt_claude, call_claude, (), ["CLAUDE_API_KEY|ANTHROPIC_API_KEY"]),
    "gemini": (build_prompt_gemini, call_gemini, ("찬성", "ok", "okay"), ["GEMINI_API_KEY"]),
    "clova": (build_prompt_clova, call_clova, (), ["CLOVA_BEARER_KEY"]),
}
