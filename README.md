# ziwei-rag

紫微斗数古籍知识库 + 命盘 AI 解读

## 结构

```
ziwei-rag/
  asset/classics/
    gusuifu.txt     ← 骨髓赋原文
    quanshu.txt     ← 紫微斗数全书节选
    quanji.txt      ← 紫微斗数全集节选
        /mingpan/   ← 放想要查询命盘的txt文件
  src/
    ingest.py       ← 建库（只跑一次）
    query.py        ← 日常使用入口
    prompt.py       ← system prompt 
  chroma_db/        ← 自动生成，向量数据库
  .env              ← 放 OPENAI_API_KEY
  requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用步骤

### 第一步：建库（只需跑一次）
```bash
python src/ingest.py
```
会下载 BAAI/bge-m3 模型，然后把古籍 embed 存入本地 ChromaDB。

### 第二步：填入你的命盘

把文墨天机给你的命盘数据直接填入asset/mingpan/里并用txt保存

### 第三步：提问
```bash
python src/query.py
```
默认按照命盘解析多角度问题

如果有特定问题，在question参数输入问题
```

## 扩充知识库

把更多古籍或资料放进 `assets/classics/` 下（.txt 格式），
重新跑 `python src/ingest.py` 即可重建。

## 模型选择

- Embed：BAAI/bge-m3（免费，本地跑，中文古文效果好）
- LLM：gpt-5