"""OpenAI-compatible chat API for NextChat.

The model is not kept in memory. Each open question spawns llama-completion and
that process exits before the HTTP response finishes, which releases VRAM.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .generate import answer_question
from .gpu_select import query_gpus, select_gpus

GPU_LOCK = threading.Lock()
URL_RE = re.compile(r"https?://[^\s|]+")
MAX_GPUS = int(os.environ.get("ASKNJUPT_MAX_GPUS", "4"))
NEED_MIB = int(os.environ.get("ASKNJUPT_GPU_NEED_MIB", "3072"))
HOST = os.environ.get("ASKNJUPT_HOST", "0.0.0.0")
PORT = int(os.environ.get("ASKNJUPT_PORT", "8787"))


def parse_refs(refs: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for part in (refs or "").split(" | "):
        part = part.strip()
        if not part:
            continue
        match = URL_RE.search(part)
        if not match:
            items.append({"title": part, "url": "", "locator": ""})
            continue
        url = match.group(0).rstrip(".,)")
        if url in seen:
            continue
        seen.add(url)
        title = part[: match.start()].strip(" |#")
        tail = part[match.end() :].strip()
        locator = tail[1:].strip() if tail.startswith("#") else tail.lstrip("#").strip()
        items.append({"title": title, "url": url, "locator": locator})
    return items


def format_reply(answer: str, refs: str) -> str:
    body = (answer or "").strip() or "资料不足，知识库里没有可回答该问题的公开页面。"
    lines = [body, "", "---", "**出处**"]
    sources = parse_refs(refs)
    if not sources:
        lines.append("本次没有定位到可引用的公开页面或文件。")
        return "\n".join(lines)
    for index, src in enumerate(sources, 1):
        title = src["title"] or src["url"] or "未命名资料"
        if src["url"]:
            lines.append(f"{index}. [{title}]({src['url']})")
        else:
            lines.append(f"{index}. {title}")
        if src["locator"]:
            lines.append(f"   定位：`{src['locator']}`")
    return "\n".join(lines)


TOPIC_MARK = "简要主题"
SUMMARY_MARK = "简要总结一下对话内容"


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for piece in content:
            if isinstance(piece, str):
                parts.append(piece)
            elif isinstance(piece, dict) and piece.get("text"):
                parts.append(piece["text"])
        return "\n".join(parts).strip()
    return ""


def short_topic(question: str) -> str:
    text = re.sub(r"[？?！!。，,\s“”\"']", "", question)
    text = re.sub(r"^\d{4}年", "", text)
    text = re.sub(r"(请问|请|是什么|是多少|多少|吗|呢)$", "", text)
    return text[:12] or "闲聊"


def utility_reply(messages: list) -> str | None:
    """NextChat asks the model to name or compress a chat. Do that without the GPU."""
    users = [message_text(message) for message in messages if isinstance(message, dict) and message.get("role") == "user"]
    users = [text for text in users if text]
    if not users:
        return None
    last = users[-1]
    prior = [text for text in users[:-1] if TOPIC_MARK not in text and SUMMARY_MARK not in text]
    if TOPIC_MARK in last or "闲聊" in last and "不要解释" in last:
        return short_topic(prior[-1] if prior else "")
    if SUMMARY_MARK in last:
        assistants = [
            message_text(message)
            for message in messages
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        assistants = [text for text in assistants if text and not text.startswith("……")]
        source = assistants[-1] if assistants else (prior[-1] if prior else "")
        source = source.split("\n---", 1)[0].strip()
        return source[:180] or "闲聊"
    return None


def last_user_text(messages: list) -> str:
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for piece in content:
                if isinstance(piece, str):
                    parts.append(piece)
                elif isinstance(piece, dict):
                    text = piece.get("text") or ""
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
    return ""


def answer_with_gpus(question: str) -> tuple[str, dict]:
    """Run one question. VRAM belongs only to the short-lived llama process."""
    with GPU_LOCK:
        chosen = select_gpus(query_gpus(), max_cards=MAX_GPUS, need_mib=NEED_MIB)
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in chosen)
        started = time.time()
        try:
            record = answer_question(question)
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
        elapsed = time.time() - started
    text = format_reply(record.get("answer") or "", record.get("refs") or "")
    meta = {
        "gpus": chosen,
        "device": "cpu" if record.get("used_llm") and not chosen else ("gpu" if record.get("used_llm") else "none"),
        "used_llm": bool(record.get("used_llm")),
        "seconds": round(elapsed, 2),
        "llm_error": record.get("llm_error"),
    }
    print(
        json.dumps({"question": question, **meta}, ensure_ascii=False),
        flush=True,
    )
    if meta["llm_error"]:
        text += f"\n\n开放题模型本次没有生成文字，已回退到资料原文。原因：{meta['llm_error']}"
    return text, meta


def _chunk(completion_id: str, content: str, *, role: str | None = None, finish: str | None = None) -> dict:
    delta: dict[str, str] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "asknjupt",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[asknjupt] {self.address_string()} {fmt % args}", flush=True)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in {"/health", "/"}:
            stats = query_gpus()
            self._json(
                200,
                {
                    "ok": True,
                    "service": "asknjupt",
                    "model_resident": False,
                    "max_gpus": MAX_GPUS,
                    "gpus": [gpu.__dict__ for gpu in stats],
                    "next_choice": select_gpus(stats, max_cards=MAX_GPUS, need_mib=NEED_MIB),
                },
            )
            return
        if path in {"/v1/models", "/models"}:
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "asknjupt",
                            "object": "model",
                            "owned_by": "asknjupt",
                        }
                    ],
                },
            )
            return
        self._json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in {"/v1/chat/completions", "/chat/completions"}:
            self._json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        length = int(self.headers.get("Content-Length") or "0")
        try:
            payload = json.loads(self.rfile.read(length).decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "invalid json", "type": "invalid_request"}})
            return
        messages = payload.get("messages") or []
        utility = utility_reply(messages)
        if utility is not None:
            text = utility
        else:
            question = last_user_text(messages)
            if not question:
                self._json(400, {"error": {"message": "messages 里没有用户问题", "type": "invalid_request"}})
                return
            try:
                text, _meta = answer_with_gpus(question)
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": {"message": str(exc), "type": "server_error"}})
                return
        completion_id = "chatcmpl-" + uuid.uuid4().hex[:12]
        if payload.get("stream"):
            self._stream(completion_id, text)
            return
        self._json(
            200,
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "asknjupt",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def _stream(self, completion_id: str, text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        self.wfile.write(("data: " + json.dumps(_chunk(completion_id, "", role="assistant"), ensure_ascii=False) + "\n\n").encode())
        step = 48
        for offset in range(0, len(text), step):
            piece = text[offset : offset + step]
            self.wfile.write(("data: " + json.dumps(_chunk(completion_id, piece), ensure_ascii=False) + "\n\n").encode())
            self.wfile.flush()
        self.wfile.write(("data: " + json.dumps(_chunk(completion_id, "", finish="stop"), ensure_ascii=False) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"asknjupt backend http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
