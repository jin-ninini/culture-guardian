"""
Chroma + OpenAIEmbeddings 벡터 DB (data/rag_repository/) 로드 및 구축/갱신.

- load_vectorstore / load_metadata : 검색 시 사용
- rebuild_from_directory          : raw_json 전체로 인덱스 재빌드
- update_from_json                : JSON 파일 하나를 기존 인덱스에 추가 (기존 인덱스는 backup/ 으로 복사)
"""

import json
import os
import pickle
import shutil
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.config import CONFIG

# .env 에서 OPENAI_API_KEY, EMBEDDING_MODEL 등을 읽는다.
load_dotenv()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def debug_print(msg: str):
    print(f"[RAG] {msg}")


def _open_chroma():
    return Chroma(
        persist_directory=CONFIG["rag_repository_path"],
        embedding_function=OpenAIEmbeddings(model=CONFIG["embedding_model"]),
    )


# ======================================================
# 로드
# ======================================================
def load_vectorstore():
    """Chroma 벡터스토어 로드"""
    repo_path = CONFIG["rag_repository_path"]
    ensure_dir(repo_path)
    vectorstore = _open_chroma()
    debug_print(f"Chroma vectorstore loaded from: {repo_path}")
    return vectorstore


def load_metadata():
    """
    rag_repository 안의 보조 파일 로드
    - vectors.pkl (없으면 자동 스킵)
    - manifest.json
    - alias_map.json
    - relation_db.csv (항상 시도)
    """
    repo_path = CONFIG["rag_repository_path"]

    metadata_path = os.path.join(repo_path, CONFIG["metadata_file"])
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            debug_print(f"metadata_file({CONFIG['metadata_file']}) 로드 완료")
        except Exception as e:
            debug_print(f"metadata_file 로드 실패: {e}")
    else:
        debug_print("metadata_file(vectors.pkl)이 없어도 계속 진행합니다.")

    manifest = load_json(os.path.join(repo_path, CONFIG["manifest_file"]), default={})
    alias_map = load_json(os.path.join(repo_path, CONFIG["alias_file"]), default={})

    relation_df = None
    relation_path = CONFIG.get("relation_db_path")
    if CONFIG.get("use_relation_db", False) and os.path.exists(relation_path):
        try:
            relation_df = pd.read_csv(relation_path)
            debug_print(f"relation_db.csv 로드 완료 ({len(relation_df)}행)")
        except Exception as e:
            debug_print(f"relation_db.csv 로드 실패: {e}")
    else:
        debug_print("relation_db.csv 사용 비활성화")

    return metadata, manifest, alias_map, relation_df


# ======================================================
# 구축 / 갱신
# ======================================================
def _load_documents(json_path: str):
    """raw JSON(문자열 또는 {text, country, topic, source} 리스트)을 Document 리스트로 변환"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if isinstance(raw_data, dict):  # 일부 파일은 단일 객체
        raw_data = [raw_data]

    docs = []
    default_country = os.path.splitext(os.path.basename(json_path))[0]
    for item in raw_data:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            meta = {"country": default_country, "topic": None, "source": os.path.basename(json_path)}
            docs.append(Document(page_content=text, metadata=meta))
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("body") or ""
            if not text:
                continue
            # 성능 평가용 문항(data/eval/eval_dataset.csv)은 인덱싱하지 않는다
            if (item.get("metadata") or {}).get("source") == "eval_dataset":
                continue
            meta = {
                "country": item.get("country") or default_country,
                "topic": item.get("topic"),
                "source": item.get("source") or os.path.basename(json_path),
            }
            docs.append(Document(page_content=text, metadata=meta))
    return docs


def _split(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    return splitter.split_documents(docs)


def _add_in_batches(vectorstore, docs, batch_size: int = 2000):
    """Chroma add_documents를 배치로 나눠 안전하게 수행."""
    for start in range(0, len(docs), batch_size):
        vectorstore.add_documents(docs[start:start + batch_size])


def _save_chunk_metadata(chunks):
    metadata_path = os.path.join(CONFIG["rag_repository_path"], CONFIG["metadata_file"])
    meta_dict = {i: {"country": c.metadata.get("country"), "source": c.metadata.get("source")}
                 for i, c in enumerate(chunks)}
    with open(metadata_path, "wb") as f:
        pickle.dump(meta_dict, f)
    debug_print(f"OK vectors.pkl 저장 완료 ({len(chunks)}개 청크)")


def _update_manifest(**fields):
    manifest_path = os.path.join(CONFIG["rag_repository_path"], CONFIG["manifest_file"])
    manifest = load_json(manifest_path, default={})
    manifest.update(fields)
    save_json(manifest_path, manifest)


def _backup_repository():
    """기존 rag_repository 전체를 backup/ 아래에 타임스탬프 폴더로 백업"""
    repo = CONFIG["rag_repository_path"]
    backup_root = CONFIG["backup_dir"]
    ensure_dir(backup_root)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_root, f"backup_{ts}")

    if not os.path.isdir(repo) or not os.listdir(repo):
        ensure_dir(backup_path)
        debug_print("백업 대상이 없어 빈 백업 폴더만 생성했습니다.")
        return ts

    def _ignore_backup(dir_path, names):
        if os.path.abspath(dir_path) == os.path.abspath(repo):
            return {"backup"}
        return set()

    shutil.copytree(repo, backup_path, dirs_exist_ok=False, ignore=_ignore_backup)
    debug_print(f"OK 기존 인덱스를 백업했습니다: {backup_path}")
    return ts


def update_from_json(json_path: str):
    """JSON 파일 하나를 받아서 RAG 인덱스를 갱신한다."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"새 RAG JSON을 찾을 수 없습니다: {json_path}")

    ts = _backup_repository()
    chunks = _split(_load_documents(json_path))
    _add_in_batches(_open_chroma(), chunks)
    debug_print("OK 새 문서를 Chroma 인덱스에 추가했습니다.")

    _save_chunk_metadata(chunks)
    _update_manifest(last_updated=ts, last_source_file=os.path.basename(json_path))
    debug_print("OK manifest.json 갱신 완료")


def rebuild_from_directory(raw_dir: str | None = None):
    """raw_json 디렉터리 전체를 읽어 인덱스를 재빌드한다."""
    raw_dir = raw_dir or CONFIG["raw_json_dir"]
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw_json 디렉터리를 찾을 수 없습니다: {raw_dir}")

    json_files = sorted(glob(os.path.join(raw_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"{raw_dir} 에 JSON 파일이 없습니다.")

    debug_print(f"총 {len(json_files)}개 JSON을 이용해 전체 재빌드합니다.")

    # 기존 인덱스 백업 후 빈 저장소 생성
    repo = CONFIG["rag_repository_path"]
    if os.path.isdir(repo) and os.listdir(repo):
        _backup_repository()
    if os.path.exists(repo):
        shutil.rmtree(repo, ignore_errors=True)
    ensure_dir(repo)

    all_chunks = []
    for jf in json_files:
        try:
            all_chunks.extend(_split(_load_documents(jf)))
        except json.JSONDecodeError as e:
            debug_print(f"JSON 파싱 실패: {jf} - {e}")

    if not all_chunks:
        raise ValueError("추출된 문서 청크가 없습니다.")

    _add_in_batches(_open_chroma(), all_chunks)
    _save_chunk_metadata(all_chunks)
    _update_manifest(
        last_updated=datetime.now().strftime("%Y%m%d_%H%M%S"),
        last_source_file=os.path.basename(raw_dir),
        num_files=len(json_files),
        num_chunks=len(all_chunks),
    )
    debug_print("OK 전체 재빌드 완료")
