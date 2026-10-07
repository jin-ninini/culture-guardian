"""
사용자 프로필(나라, 인종 등) + relation_db.csv + RAG 인덱스를 종합해서
최종 점수가 높은 문서부터 리턴하는 모듈

핵심 요약:
- relation_db.csv는 "가중치"가 아닌, 답변 전 항상 참고해야 하는 맥락(context) 데이터로 사용.
- country_alias.json 기반으로 질문/DB 모두 다국어 정규화.
- 질문 내 국가 감지 → 해당 국가 기반 RAG 필터링.
- relation_db도 한국어 명칭 기준이지만, alias를 통해 글로벌 비교.
"""

import os
import json
from typing import Dict, List, Any
import pandas as pd

from src.rag.config import CONFIG
from src.rag.vector_store import load_vectorstore, load_metadata, debug_print

COUNTRY_ALIAS_PATH = CONFIG["country_alias_path"]

# 전역 변수 (최적화를 위해 캐싱)
COUNTRY_ALIAS_CACHE = None


def _load_country_alias() -> dict:
    """country_alias.json 로드 (캐싱 지원)"""
    global COUNTRY_ALIAS_CACHE
    if COUNTRY_ALIAS_CACHE is not None:
        return COUNTRY_ALIAS_CACHE

    if not os.path.exists(COUNTRY_ALIAS_PATH):
        debug_print("country_alias.json 파일을 찾을 수 없습니다.")
        COUNTRY_ALIAS_CACHE = {}
        return COUNTRY_ALIAS_CACHE

    try:
        with open(COUNTRY_ALIAS_PATH, "r", encoding="utf-8") as f:
            COUNTRY_ALIAS_CACHE = json.load(f)
        debug_print(f"country_alias.json 로드 완료 ({len(COUNTRY_ALIAS_CACHE)}개 국가)")
    except Exception as e:
        debug_print(f"country_alias.json 로드 실패: {e}")
        COUNTRY_ALIAS_CACHE = {}
    return COUNTRY_ALIAS_CACHE


def normalize_country_alias(name: str) -> str:
    """
    다국어 국가명을 canonical 국가명(영문 표준)으로 정규화.
    예: "중국" / "China" / "中国" → "China"
    """
    if not name:
        return ""
    name = name.strip().lower()
    aliases = _load_country_alias()
    for canonical, alias_list in aliases.items():
        if not isinstance(alias_list, list):
            continue
        if name in [a.lower() for a in alias_list]:
            return canonical
    return name.capitalize()  # fallback


def detect_country_from_query(query: str) -> str | None:
    """
    country_alias.json을 이용해 질문에서 나라 이름 감지.
    다국어 대응 (영문, 한글, 일본어, 중국어 등)
    """
    aliases = _load_country_alias()
    q_lower = query.lower()
    for canonical, alias_list in aliases.items():
        for alias in alias_list:
            if alias.lower() in q_lower:
                debug_print(f"질문에서 감지된 나라: {canonical}")
                return canonical
    return None


def _calc_final_score(similarity: float, country_match: float) -> float:
    """간단한 점수 계산식"""
    return (
        CONFIG["similarity_weight"] * similarity
        + CONFIG["country_match_weight"] * country_match
    )


def _get_relation_context(rel_df: pd.DataFrame, user_country: str, query: str) -> str:
    """
    relation_db.csv에서 user-country, target-country에 맞는 관계 설명(txt)을 찾아 반환.
    모든 비교는 country_alias.json을 통한 정규화 기반으로 수행.
    """
    if rel_df is None or rel_df.empty:
        return ""

    aliases = _load_country_alias()
    uc_norm = normalize_country_alias(user_country)
    query_lower = query.lower()

    # 질문에 포함된 나라(다국어) → canonical 정규화
    detected_in_query = detect_country_from_query(query)

    for _, row in rel_df.iterrows():
        user = normalize_country_alias(str(row.get("user", "")).strip())
        target = normalize_country_alias(str(row.get("target", "")).strip())
        relation = str(row.get("relation", "")).strip()
        txt = str(row.get("txt", "")).strip()

        if not user or not target or not txt:
            continue

        # 유저가 user-country에 해당 + 질문에 target-country 언급
        if user == uc_norm and (detected_in_query == target or target.lower() in query_lower):
            return f"[관계 참고] {user}-{target} ({relation}): {txt}"

    return ""


def search_with_priority(query: str, user_profile: Dict[str, Any], k: int = None) -> List[Dict[str, Any]]:
    """
    RAG 검색 + relation_db.csv의 맥락 문장 결합 + 나라 감지 필터링
    """
    if k is None:
        k = CONFIG["top_k"]

    user_country = user_profile.get("country", "")
    user_country_norm = normalize_country_alias(user_country)

    # 1️⃣ 인덱스 및 메타데이터 로드
    vectorstore = load_vectorstore()
    metadata, manifest, alias_map, relation_df = load_metadata()

    # 2️⃣ relation_db.csv 기반 context 문장 추출
    relation_context = ""
    if CONFIG.get("relation_as_context", False):
        relation_context = _get_relation_context(relation_df, user_country, query)

    # 3️⃣ 질문에서 나라 자동 감지
    detected_country = detect_country_from_query(query)
    where_filter = None
    if detected_country:
        where_filter = {"country": {"$in": [detected_country]}}
        debug_print(f"질문 내 감지된 나라 기반 필터 적용: {detected_country}")
    elif user_country_norm:
        where_filter = {"country": {"$in": [user_country_norm]}}

    # 4️⃣ RAG 검색 수행
    raw_results = vectorstore.similarity_search_with_score(
        query,
        k=k,
        filter=where_filter,
    )

    ranked = []
    for doc, sim_score in raw_results:
        similarity = 1.0 - sim_score if sim_score <= 1 else sim_score
        target_country = normalize_country_alias(doc.metadata.get("country", ""))
        country_match = 1.0 if target_country == user_country_norm else 0.0
        final_score = _calc_final_score(similarity, country_match)

        ranked.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": round(final_score, 4),
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    # 5️⃣ 관계 문장은 항상 맨 앞에 추가 (GPT 프롬프트용)
    if relation_context:
        ranked.insert(0, {
            "text": relation_context,
            "metadata": {"source": "relation_db.csv", "type": "relation_context"},
            "score": 1.0,
        })

    debug_print(f"query='{query}' → {len(ranked)}개 결과 반환 (relation_context 포함)")
    return ranked
