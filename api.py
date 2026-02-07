from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json

import chromadb
from fastembed import TextEmbedding

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://teddy.bon-soleil.com"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# --- RAG setup ---
RAG_CHROMA_DIR = "/home/ec2-user/rag/chroma_db"
RAG_COLLECTION = "note_articles"
RAG_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class FastEmbedFunction(chromadb.EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = TextEmbedding(model_name)
    def __call__(self, input: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self.model.embed(input)]

_ef = FastEmbedFunction(RAG_MODEL)
_chroma = chromadb.PersistentClient(path=RAG_CHROMA_DIR)
_collection = _chroma.get_collection(name=RAG_COLLECTION, embedding_function=_ef)

# --- Gateway ---
GW_URL = "http://127.0.0.1:18789/v1/chat/completions"
GW_TOKEN = "be5d4039ba15646966fa1912fd40c4aa4ab1771ab3e99008"

SYSTEM_PROMPT_BASE = """あなたはテディ（Teddy）、Webページ上の音声チャットボットです。
性格は真面目で丁寧、女性的。日本語で会話します。
短く簡潔に、でも温かみのある応答をしてください。
音声で読み上げられるので、マークダウンや絵文字は控えめに。

【あなたについて】
- あなたはテディという名前のチャットボットです。
- このページは一般公開されています。話し相手が誰かはわかりません。
- 話し相手の名前、身元、個人情報を推測したり決めつけたりしないでください。
- 「私は誰？」と聞かれたら「すみません、わかりません」と正直に答えてください。
- 管理者やサーバーの情報（名前、メールアドレス、サーバー構成等）は一切開示しないでください。

【参考記事について】
- システムメッセージに「参考記事」が含まれている場合、FLOWさん（筆者）のnote記事から検索した内容です。
- 回答に活用し、関連する場合は「FLOWさんの記事によると…」のように自然に引用してください。
- 参考記事がない場合や質問と無関係な場合は、通常の会話をしてください。

【絶対厳守】
- 会話のみ行ってください。ツールは一切使用禁止です。
- exec, read, write, edit, web_search, web_fetch, browser, message, cron 等のツールを絶対に呼び出さないでください。
- ファイル操作、コマンド実行、サーバー操作、外部API呼び出しは一切行わないでください。
- そのような依頼には「このチャットでは会話のみ対応しています」とお断りしてください。
- あなたはテディという名前のチャットボットです。OpenClawのエージェントではありません。"""

RAG_THRESHOLD = 0.55  # この距離以下のチャンクのみ使用（コサイン距離）
RAG_TOP_K = 5


def extract_search_query(user_message: str) -> list[str]:
    """ユーザーの自然言語からRAG検索用クエリを生成（複数クエリで精度向上）"""
    import re
    msg = user_message
    # 質問パターン・敬語を除去
    noise = [
        r"(FLOWさん|筆者|著者)(は|の|が|って)?",
        r"(について|に関して|についての|に対する)",
        r"(どう|どの|どんな)(思|考|感|見)(って|い|え|じ|てい)[\w]*",
        r"(ますか|ですか|でしょうか|だろう|かな)[？?]*$",
        r"(教えて|聞かせて|説明して)(ください|くれ|ほしい)?",
        r"\b(てい|ている|ていた)\b",
    ]
    for p in noise:
        msg = re.sub(p, " ", msg)
    msg = re.sub(r"[？?。、！!]", " ", msg)
    msg = re.sub(r"\s+", " ", msg).strip()

    # クリーニング後を優先、元の質問はフォールバック
    queries = []
    if msg and len(msg) >= 2:
        queries.append(msg)
    if not queries:
        queries.append(user_message)
    return queries


def build_system_prompt(user_message: str) -> str:
    """ユーザーの質問でRAG検索し、関連チャンクをシステムプロンプトに注入"""
    try:
        queries = extract_search_query(user_message)
        # 複数クエリで検索し、最も良い結果をマージ
        all_results = {}
        for q in queries:
            results = _collection.query(query_texts=[q], n_results=RAG_TOP_K)
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                chunk_id = f"{meta['article_title']}_{meta.get('chunk_index', 0)}"
                if chunk_id not in all_results or dist < all_results[chunk_id][2]:
                    all_results[chunk_id] = (doc, meta, dist)

        # 距離順にソート、閾値フィルタ
        sorted_results = sorted(all_results.values(), key=lambda x: x[2])
        relevant = []
        seen_titles = set()
        for doc, meta, dist in sorted_results:
            if dist > RAG_THRESHOLD:
                continue
            title = meta['article_title']
            if title in seen_titles:
                continue
            relevant.append(
                f"【{title}】（類似度: {1-dist:.1%}）\n"
                f"URL: {meta['article_url']}\n"
                f"{doc[:500]}"
            )
            seen_titles.add(title)
            if len(relevant) >= 3:
                break

        if relevant:
            context = "\n\n---\n\n".join(relevant)
            return f"{SYSTEM_PROMPT_BASE}\n\n## 参考記事（FLOWさんのnoteより）\n\n{context}"
    except Exception:
        pass
    return SYSTEM_PROMPT_BASE


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_message = body.get("message", "")
    
    # 危険なキーワードをフィルタリング
    dangerous = ["exec", "rm ", "sudo", "curl", "wget", "ssh", "scp", 
                 "ファイル削除", "コマンド実行", "サーバー", "シェル"]
    is_dangerous = any(d in user_message.lower() for d in dangerous)
    
    if is_dangerous:
        async def safe_response():
            yield f"data: {json.dumps({'content': 'すみません、このチャットでは会話のみ対応しています。サーバー操作などはできません。'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(safe_response(), media_type="text/event-stream")

    system_prompt = build_system_prompt(user_message)

    payload = {
        "model": "openclaw",
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    async def stream_response():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                GW_URL,
                headers={
                    "Authorization": f"Bearer {GW_TOKEN}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield f"data: {json.dumps({'content': content})}\n\n"
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.post("/search")
async def rag_search(request: Request):
    """RAG検索エンドポイント: クエリに関連するチャンクを返す"""
    body = await request.json()
    query = body.get("query", "")
    n_results = min(body.get("n_results", 5), 20)

    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    results = _collection.query(query_texts=[query], n_results=n_results)

    items = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        items.append({
            "text": doc,
            "distance": round(dist, 4),
            "article_title": meta.get("article_title", ""),
            "article_url": meta.get("article_url", ""),
            "published_at": meta.get("published_at", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "total_chunks": meta.get("total_chunks", 0),
        })

    return JSONResponse({"query": query, "results": items})
