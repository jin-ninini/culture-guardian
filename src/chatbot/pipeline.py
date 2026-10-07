"""Answer flow: intent filter → GPT secondary check → memory → RAG → persona prompt → GPT answer."""
from src.chatbot.persona import get_persona_prompt
from src.chatbot.preference_store import PreferenceStore
from src.chatbot.safety import CHAT_MODEL, get_openai_client, get_sensitivity_note, gpt_secondary_intent_check
from src.chatbot.user_store import append_memory
from src.intent_classifier.inference import predict_intent
from src.rag.retriever import search_with_priority


def answer_question(question, profile, intent_model):
    """Return the assistant reply for one question. intent_model = load_intent_model(...) result."""
    # Intent
    try:
        tokenizer, model, device, cfg = intent_model
        intent_res = predict_intent(question, tokenizer, model, device, cfg)
        prob = intent_res["probabilities"]
    except Exception:
        prob = {}

    # GPT secondary
    if prob.get("비하/도발", 0.0) >= 0.50:
        sec = gpt_secondary_intent_check(question)
        if sec.get("judgement") == "unsafe":
            return (
                f"이 표현은 오해될 수 있어요.\n"
                f"이유: {sec.get('reason')}\n"
                f"더 자연스러운 표현: {sec.get('rephrased')}"
            )

    # Memory update
    pref = PreferenceStore(profile["user_id"])
    pref.update_memory_slots(question)
    memory_prompt = pref.get_memory_prompt()

    # 민감성 힌트
    sensitivity_note = get_sensitivity_note(memory_prompt, question)

    # RAG
    try:
        results = search_with_priority(question, {"country": profile["country"]})
        context = "\n".join(r["text"] for r in results)
    except Exception:
        context = ""

    # Persona
    persona = get_persona_prompt(profile)

    # Prompt
    full_prompt = (
        f"{memory_prompt}\n\n"
        f"### 사용자 질문:\n{question}\n\n"
        f"### 참고 문맥(RAG):\n{context}\n\n"
        f"### 추가 참고사항 (민감성 힌트):\n{sensitivity_note}\n\n"
        "### 답변 지침:\n"
        "- 결론을 먼저 말하는 두괄식으로 답변합니다.\n"
        "- 사용자의 질문 의도가 선물/식사/예절 고민이라면, 상대를 존중하는 방향으로 안내합니다.\n"
        "- 만약 위의 민감성 힌트가 주어진 경우, "
        "치킨·수박 등은 '좋은 선택이지만', '괜찮은 선택일 수 있지만'처럼 먼저 긍정 평가하지 말고, "
        "'그 선택은 조금 조심하는 편이 좋아요'처럼 부드럽게 조심을 권하는 톤으로 설명하세요.\n"
        "- '특정 인종' 같은 모호한 표현 대신, 이미 주어진 상대 정보(흑인 친구 등)를 자연스럽게 사용합니다.\n"
        "- 과도한 역사 강의처럼 길게 설명하지 말고, 이유는 짧게 정리하고 바로 대안을 제안합니다.\n"
        "- 문화적 금기 또는 오해 소지가 있는 경우 절대 긍정 표현으로 시작하지 말 것.\n"
        "- 금기 상황이면 곧바로 “그 선택은 피하는 것이 좋아요”라고 단정적으로 시작.\n"
        "- 이유는 짧고 단정적으로.\n"
        "- 이어서 바로 대안을 제안할 것.\n"
        "- 금기가 아닌 일반 상황에서는 평소처럼 부드럽고 친근한 톤 유지.\n\n"
        "### 답변:"
    )

    # LLM
    try:
        resp = get_openai_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": persona},
                {"role": "user", "content": full_prompt},
            ]
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"(오류 발생) {e}"

    # 저장
    append_memory(profile["user_id"], question, answer)
    pref.extract_prompt_summary()

    return answer
