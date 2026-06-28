"""
ingest.py
读取 assets/classics/ 下所有 .txt 古籍，
按段落 chunk，用 BAAI/bge-m3 embed，存入本地 ChromaDB。
运行一次即可，之后直接用 query.py。
"""
# export HF_ENDPOINT=https://hf-mirror.com

import os
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

CLASSICS_DIR = Path(__file__).parent.parent / "asset" / "classics"
CHROMA_DIR   = Path(__file__).parent.parent / "chroma_db"
COLLECTION   = "ziwei_classics"
EMBED_MODEL  = "BAAI/bge-m3"  # 免费，中文古文效果好


def load_and_chunk(filepath: Path) -> list[dict]:
    """
    把一个 .txt 古籍文件切成 chunk。
    切法：按空行分段，每段保留来源文件名和段落序号。
    对于歌诀类（骨髓赋每条断语），保持每条独立。
    """
    text = filepath.read_text(encoding="utf-8")
    source = filepath.stem  # e.g. "gusuifu"

    # 去掉注释行（# 开头）
    lines = [l for l in text.splitlines() if not l.startswith("#")]
    text = "\n".join(lines)

    # 按双换行或章节分隔符切块
    raw_chunks = re.split(r"\n{2,}|(?=={2,})", text)

    chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if len(chunk) < 10:  # 过滤太短的片段
            continue
        chunks.append({
            "id":     f"{source}_{i:04d}",
            "text":   chunk,
            "source": source,
            "idx":    i,
        })

    return chunks


def build_db():
    print(f"[1/4] 加载 embed 模型：{EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("[2/4] 读取古籍文件并 chunk...")
    all_chunks = []
    for txt_file in sorted(CLASSICS_DIR.glob("*.txt")):
        chunks = load_and_chunk(txt_file)
        print(f"  {txt_file.name}: {len(chunks)} 个 chunk")
        all_chunks.extend(chunks)
    print(f"  合计：{len(all_chunks)} 个 chunk")

    print("[3/4] 生成 embeddings（首次较慢）...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # 余弦相似度用
    )

    print("[4/4] 存入 ChromaDB...")
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 如果已有同名 collection 先删掉重建
    try:
        client.delete_collection(COLLECTION)
        print("  已删除旧 collection，重新构建")
    except Exception:
        pass

    col = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    col.add(
        ids=[c["id"] for c in all_chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"source": c["source"], "idx": c["idx"]} for c in all_chunks],
    )

    print(f"\n完成！共存入 {len(all_chunks)} 条，数据库在 {CHROMA_DIR}")


if __name__ == "__main__":
    build_db()
