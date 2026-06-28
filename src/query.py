"""
query.py
主入口：从 asset/mingpan/ 读取命盘 + 可选问题，RAG 解读，结果写入 output/。
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

from prompt import SYSTEM_PROMPT

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

CHROMA_DIR   = ROOT / "chroma_db"
MINGPAN_DIR  = ROOT / "asset" / "mingpan"
OUTPUT_DIR   = ROOT / "output"
DEFAULT_MINGPAN = MINGPAN_DIR / "test.txt"
COLLECTION  = "ziwei_classics"
EMBED_MODEL = "BAAI/bge-m3"
TOP_K       = 10


# ── 初始化（只加载一次，避免每次调用都重新加载模型）────────────────────────

_embed_model = None
_chroma_col  = None

def _get_resources():
    global _embed_model, _chroma_col
    if _embed_model is None:
        print("加载 embed 模型（首次较慢）...")
        _embed_model = SentenceTransformer(EMBED_MODEL)
    if _chroma_col is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_col = client.get_collection(COLLECTION)
    return _embed_model, _chroma_col


# ── 检索古籍 ─────────────────────────────────────────────────────────────────

def retrieve(query: str, model, col) -> list[str]:
    vec = model.encode([query], normalize_embeddings=True)
    results = col.query(
        query_embeddings=vec.tolist(),
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )
    chunks = []
    label_map = {
        "gusuifu": "《骨髓赋》",
        "quanshu": "《紫微斗数全书》",
        "quanji":  "《紫微斗数全集》",
    }
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), start=1
    ):
        label = label_map.get(meta["source"], meta["source"])
        chunks.append(f"[{i}] {label}\n{doc}")
    return chunks


def _build_retrieval_query(raw_text: str, question: str = "") -> str:
    """用问题 + 命盘原文片段做向量检索，保留星曜名称等关键信息。"""
    excerpt = raw_text.strip()
    if len(excerpt) > 1500:
        excerpt = excerpt[:1500]
    question = question.strip()
    if question:
        return f"{question}\n{excerpt}"
    return excerpt


# ── 解读 ─────────────────────────────────────────────────────────────────────

def interpret(raw_text: str, question: str, openai_client: OpenAI) -> str:
    model, col = _get_resources()

    retrieval_query = _build_retrieval_query(raw_text, question)
    chunks    = retrieve(retrieval_query, model, col)
    knowledge = "\n\n---\n\n".join(chunks)

    user_message = f"""## 原始命盘

{raw_text.strip()}

## 参考古籍段落（系统检索，编号 [1]–[{len(chunks)}]，相关度由高到低）

解读时须引用这些编号作为 citation，格式如 [1]、[2][3]。

{knowledge}
"""
    question = question.strip()
    if question:
        user_message += f"""
---

## 问题

{question}
"""

    response = openai_client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=1,
    )
    return response.choices[0].message.content


def load_mingpan(path: Path) -> str:
    """从 asset/mingpan/ 或任意路径读取命盘文本。"""
    p = path if path.is_absolute() else ROOT / path
    return p.read_text(encoding="utf-8")


def save_output(content: str, mingpan_path: Path, output_path: Path = None) -> Path:
    """将解读结果写入 output/，默认文件名为 {命盘名}_{时间戳}.txt。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"{mingpan_path.stem}_{timestamp}.txt"
    else:
        output_path = output_path if output_path.is_absolute() else ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


# ── 主入口 ───────────────────────────────────────────────────────────────────

def ask(raw_text: str, question: str = "") -> str:
    """
    唯一的对外接口。
    raw_text : 从文墨天机复制的原始命盘文本（任意格式）
    question : 可选。留空则按 system prompt 做整体解读；有内容时追加针对性问题
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("检索古籍 + 解读...")
    return interpret(raw_text, question, client)


# ── 使用 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="紫微斗数 RAG 命盘解读")
    parser.add_argument(
        "--mingpan",
        type=Path,
        default=DEFAULT_MINGPAN,
        help=f"命盘文件路径（默认 {DEFAULT_MINGPAN.relative_to(ROOT)}）",
    )
    parser.add_argument(
        "--question",
        default="",
        help="可选的针对性问题；留空则做整体解读",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出文件路径（默认 output/{命盘名}_{时间戳}.txt）",
    )
    args = parser.parse_args()

    mingpan_path = args.mingpan if args.mingpan.is_absolute() else ROOT / args.mingpan
    raw_text = load_mingpan(mingpan_path)
    result = ask(raw_text, args.question)

    out_path = save_output(result, mingpan_path, args.output)
    print(f"\n已保存至 {out_path}")
    print("\n" + "=" * 60)
    print(result)
