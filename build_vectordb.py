"""Build the RAG vector DB (data/rag_repository/) from data/rag/raw_json/ with OpenAI embeddings.

Without arguments the whole index is rebuilt. With --add, one JSON file is appended
to the existing index (the previous index is copied to data/rag_repository_backup/).
"""
import argparse

from src.rag.vector_store import rebuild_from_directory, update_from_json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=str, default=None, help="directory of raw JSON files (default: data/rag/raw_json)")
    parser.add_argument("--add", type=str, default=None, metavar="JSON", help="append one JSON file instead of rebuilding")
    args = parser.parse_args()

    if args.add:
        update_from_json(args.add)
    else:
        rebuild_from_directory(args.raw_dir)


if __name__ == "__main__":
    main()
