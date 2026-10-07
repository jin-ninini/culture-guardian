"""File-based user accounts: data/users/<user_id>/{profile.json, memory.jsonl, preferences.json}."""
import hashlib
import hmac
import json
import os

from src.config import USERS_DIR


def _hash_password(pw: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, 200_000).hex()


def register_user(user_id, pw, race, country):
    """Create a user folder. Returns (success, message)."""
    user_dir = os.path.join(USERS_DIR, user_id)

    if os.path.exists(user_dir):
        return False, "이미 존재하는 아이디입니다."

    os.makedirs(user_dir, exist_ok=True)

    salt = os.urandom(16)
    with open(os.path.join(user_dir, "profile.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"user_id": user_id, "password_salt": salt.hex(), "password_hash": _hash_password(pw, salt),
             "race": race, "country": country},
            f, ensure_ascii=False, indent=2
        )

    open(os.path.join(user_dir, "memory.jsonl"), "w").close()

    return True, "회원가입 완료! 로그인해주세요!"


def login_user(user_id, pw):
    """Returns (profile or None, message). The returned profile has no password fields."""
    path = os.path.join(USERS_DIR, user_id, "profile.json")

    if not os.path.exists(path):
        return None, "존재하지 않는 아이디입니다."

    with open(path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    expected = profile.pop("password_hash")
    salt = bytes.fromhex(profile.pop("password_salt"))
    if not hmac.compare_digest(_hash_password(pw, salt), expected):
        return None, "비밀번호가 틀렸습니다."

    return profile, f"환영합니다, {user_id}님!"


def append_memory(user_id, question, answer):
    with open(os.path.join(USERS_DIR, user_id, "memory.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"user": question, "bot": answer}, ensure_ascii=False) + "\n")
