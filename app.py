"""URTH : Culture Guardian — Gradio chat app (intent filter + RAG + GPT answer)."""
import argparse
import html
import os

import gradio as gr

from src.chatbot.pipeline import answer_question
from src.chatbot.user_store import login_user, register_user
from src.config import ASSETS_DIR, DEFAULT_INTENT_CONFIG
from src.intent_classifier.inference import load_intent_model

URTH_ICON = os.path.join(ASSETS_DIR, "URTH.png")
SPACE_BG = os.path.join(ASSETS_DIR, "space.png")

with open(os.path.join(ASSETS_DIR, "style.css"), encoding="utf-8") as f:
    CGW_CSS = f.read().replace("{SPACE_BG}", SPACE_BG)

# main()에서 load_intent_model() 결과로 채워짐
INTENT_MODEL = None


# =========================================
# 렌더링 함수
# =========================================
def render_history(history):
    rows = []
    for role, text in history:
        safe = html.escape(text)

        if role == "user":
            rows.append(
                f"""
                <div class="chat-row user">
                    <div class="chat-bubble user">{safe}</div>
                </div>
                """
            )
        else:
            rows.append(
                f"""
                <div class="chat-row bot">
                    <div class="chat-avatar-img">
                        <img src="/gradio_api/file={URTH_ICON}" />
                    </div>
                    <div class="chat-bubble bot">{safe}</div>
                </div>
                """
            )

    return f'<div class="chat-window">{"".join(rows)}</div>'


# =========================================
# 회원가입
# =========================================
def register_user_web(user_id, pw, race, country):
    ok, msg = register_user(user_id, pw, race, country)
    if not ok:
        return msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    return msg, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


# =========================================
# 로그인
# =========================================
def login_web(user_id, pw):
    profile, msg = login_user(user_id, pw)
    if profile is None:
        return None, msg, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    return profile, msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


# =========================================
# Chat 처리
# =========================================
def chat_process(question, profile, history):
    if history is None:
        history = []

    question = (question or "").strip()
    if not question:
        return history, render_history(history)

    history.append(("user", question))

    if profile is None:
        history.append(("assistant", "먼저 로그인해주세요."))
        return history, render_history(history)

    history.append(("assistant", answer_question(question, profile, INTENT_MODEL)))
    return history, render_history(history)


# =========================================
# UI 구성
# =========================================
with gr.Blocks(title="URTH : Culture Guardian") as demo:

    state_profile = gr.State()
    state_history = gr.State([])

    gr.HTML("""
        <div class="cgw-title">
            <h1>URTH : CULTURE GUARDIAN</h1>
            <p>다른 문화권 친구들과의 대화를 URTH가 도와줘요.</p>
        </div>
    """)

    # -------------------------------
    # 로그인 페이지
    # -------------------------------
    login_page = gr.Group(visible=True)
    with login_page:
        with gr.Column(elem_classes=["cgw-card"]):
            uid = gr.Textbox(label="아이디")
            pw = gr.Textbox(label="비밀번호", type="password")
            login_btn = gr.Button("로그인", variant="primary")
            login_msg = gr.Textbox(label="알림", interactive=False)
            go_signup = gr.Button("회원가입", variant="secondary")

    # -------------------------------
    # 회원가입 페이지
    # -------------------------------
    signup_page = gr.Group(visible=False)
    with signup_page:
        with gr.Column(elem_classes=["cgw-card"]):
            rid = gr.Textbox(label="아이디")
            rpw = gr.Textbox(label="비밀번호", type="password")
            race = gr.Textbox(label="인종")
            country = gr.Textbox(label="국가")
            signup_btn = gr.Button("회원가입 완료", variant="primary")
            signup_msg = gr.Textbox(label="알림", interactive=False)
            back_login = gr.Button("⬅ 로그인으로 돌아가기", variant="secondary")

    # -------------------------------
    # 채팅 페이지
    # -------------------------------
    chat_page = gr.Group(visible=False)
    with chat_page:
        with gr.Column(elem_classes=["cgw-card"]):
            chat_html = gr.HTML(value=render_history([]))
            qbox = gr.Textbox(label="질문 입력")
            send_btn = gr.Button("전송", variant="primary")
            logout_btn = gr.Button("로그아웃", variant="secondary")

    # 로그인 실행
    login_btn.click(
        login_web,
        [uid, pw],
        [state_profile, login_msg, login_page, signup_page, chat_page]
    )

    pw.submit(
        login_web,
        [uid, pw],
        [state_profile, login_msg, login_page, signup_page, chat_page]
    )

    # 화면 전환: 로그인 → 회원가입
    go_signup.click(
        lambda: (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False)
        ),
        None, [login_page, signup_page, chat_page]
    )

    # 회원가입 실행 (버튼)
    signup_btn.click(
        register_user_web,
        [rid, rpw, race, country],
        [signup_msg, login_page, signup_page, chat_page]
    )

    # 회원가입 Enter 실행
    country.submit(
        register_user_web,
        [rid, rpw, race, country],
        [signup_msg, login_page, signup_page, chat_page]
    )

    # 화면 전환: 회원가입 → 로그인
    back_login.click(
        lambda: (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        ),
        None, [login_page, signup_page, chat_page]
    )

    # 채팅 전송
    send_btn.click(
        chat_process,
        [qbox, state_profile, state_history],
        [state_history, chat_html]
    ).then(lambda: "", None, [qbox])

    qbox.submit(
        chat_process,
        [qbox, state_profile, state_history],
        [state_history, chat_html]
    ).then(lambda: "", None, [qbox])

    # --------------------------------------------
    # 로그아웃 기능 (상태 + 입력값 + 채팅창 전체 초기화)
    # --------------------------------------------
    def logout_fn():
        return (
            None,                  # state_profile clear
            [],                    # state_history clear
            gr.update(visible=True),   # login page on
            gr.update(visible=False),  # signup page off
            gr.update(visible=False),  # chat page off
            "",                     # uid clear
            "",                     # pw clear
            "",                     # rid clear
            "",                     # rpw clear
            "",                     # race clear
            "",                     # country clear
            "",                     # qbox clear
            render_history([]),     # chat_html 초기화
            "",                     # login_msg clear
            ""                      # signup_msg clear
        )

    logout_btn.click(
        logout_fn,
        None,
        [
            state_profile,
            state_history,
            login_page,
            signup_page,
            chat_page,
            uid,
            pw,
            rid,
            rpw,
            race,
            country,
            qbox,
            chat_html,
            login_msg,
            signup_msg,
        ]
    )



def main():
    global INTENT_MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=DEFAULT_INTENT_CONFIG, help="intent classifier config")
    parser.add_argument("--no-share", action="store_true", help="do not create a public gradio.live link")
    args = parser.parse_args()

    INTENT_MODEL = load_intent_model(args.config)
    demo.launch(share=not args.no_share, css=CGW_CSS, allowed_paths=[ASSETS_DIR])


if __name__ == "__main__":
    main()
