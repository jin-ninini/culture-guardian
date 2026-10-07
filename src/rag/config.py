import os

from dotenv import load_dotenv

from src.config import DATA_DIR

load_dotenv()

RAG_DATA_DIR = os.path.join(DATA_DIR, "rag")

CONFIG = {
    # 임베딩 모델 (OpenAI) — .env에 빈 값이면 기본값 사용
    "embedding_model": os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small",

    # RAG 저장소 위치 (build_vectordb.py가 생성하는 벡터 DB)
    "rag_repository_path": os.path.join(DATA_DIR, "rag_repository"),

    # 원천 데이터(JSON) 위치
    "raw_json_dir": os.path.join(RAG_DATA_DIR, "raw_json"),

    # 관계 DB 위치 (답변 컨텍스트로 사용)
    "relation_db_path": os.path.join(RAG_DATA_DIR, "relation_db.csv"),

    # 다국어 국가명 alias
    "country_alias_path": os.path.join(RAG_DATA_DIR, "country_alias.json"),

    # 메타/보조 파일 이름 (rag_repository 안에 있어야 함)
    "metadata_file": "vectors.pkl",
    "manifest_file": "manifest.json",
    "alias_file": "alias_map.json",

    # 검색 파라미터
    "top_k": 5,
    "similarity_weight": 0.8,
    "country_match_weight": 0.2,

    # 백업 디렉터리 (재빌드 시 rag_repository를 지우므로 그 바깥에 둔다)
    "backup_dir": os.path.join(DATA_DIR, "rag_repository_backup"),

    # 관계 DB를 항상 답변 맥락에 포함시킬지 여부
    "use_relation_db": True,
    "relation_as_context": True,
}
