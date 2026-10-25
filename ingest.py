"""Bulk-load files from kb/ into ONE server's knowledge base (admins can also use /kb upload).

Usage:
    python ingest.py <guild_id>            # add / update documents
    python ingest.py <guild_id> --reset    # wipe that server's knowledge base first
"""
import sys
from pathlib import Path

from dotenv import load_dotenv
from upstash_vector import Index, Vector

load_dotenv()
index = Index.from_env()


def chunk(text: str, size: int = 900) -> list[str]:
    pieces = []
    for para in (p.strip() for p in text.split("\n\n")):
        while len(para) > size:  # one huge paragraph: cut it at a space so no chunk is oversized
            cut = para.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            pieces.append(para)
    chunks, current = [], ""
    for para in pieces:
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


def main():
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        sys.exit("usage: python ingest.py <guild_id> [--reset]")
    namespace = f"kb-{sys.argv[1]}"
    if "--reset" in sys.argv:
        index.reset(namespace=namespace)
        print("knowledge base cleared")
    vectors = []
    for path in sorted(Path("kb").glob("*")):
        if path.suffix not in {".md", ".txt"}:
            continue
        for i, text in enumerate(chunk(path.read_text(encoding="utf-8"))):
            vectors.append(
                Vector(id=f"{path.stem}-{i}", data=text, metadata={"text": text, "source": path.name})
            )
    for start in range(0, len(vectors), 50):
        index.upsert(vectors=vectors[start : start + 50], namespace=namespace)
    print(f"uploaded {len(vectors)} chunks to {namespace}")


if __name__ == "__main__":
    main()
