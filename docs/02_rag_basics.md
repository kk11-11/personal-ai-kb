# RAG 基础笔记

## 什么是 RAG
RAG = Retrieval-Augmented Generation。给 LLM "外挂"一个知识库:
提问 → 从知识库检索相关片段 → 把片段和问题一起塞给 LLM → 让 LLM 基于资料回答。

## 为什么需要 RAG
- LLM 知识有截止日期,私人 / 最新资料它不知道。
- 直接让 LLM 答会"幻觉"(瞎编)。RAG 让它有依据。
- 比微调便宜、随时更新。

## 关键步骤
1. 文档解析:txt / md 直接读,PDF 用 pdfplumber / PyPDF2。
2. 切分(chunking):太长塞不进 prompt,太短丢上下文。一般 300–800 字符 + overlap。
3. 向量化(embedding):用 sentence-transformers 把每个 chunk 变成向量。
4. 存入向量数据库:Chroma / FAISS / Milvus 等。
5. 检索:问题也变向量,找最相似的 k 个 chunk(top-k)。
6. 生成 prompt:把检索到的 chunk + 问题拼好,调用 LLM。
7. LLM 回答,并标注引用来源。

## 常见坑
- chunk 切得太大:塞不下、检索不准。
- chunk 切得太小:上下文断裂、答案不连贯。
- 没用 rerank:top-k 里有不相关片段会干扰 LLM。
- prompt 没让 LLM "不知道就说不知道":会编。
