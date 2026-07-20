"""
OpenAI-compatible inference server using HuggingFace transformers.
Usage: python simple_server.py --model D:\models\Qwen3.5-2B --port 8000
"""
import argparse
import asyncio
import json
import time
import uuid
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

app = FastAPI()
model = None
tokenizer = None
model_name = "Qwen3.5-2B"


def load_model(model_path: str, device: str = "cuda"):
    global model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map=device if device == "cuda" else device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded. Device: {model.device}")
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")


def apply_chat_template(messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return "\n".join(m.get("content", "") for m in messages)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.0)
    stream = body.get("stream", False)

    prompt = apply_chat_template(messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    prompt_tokens = inputs.input_ids.shape[1]
    print(f"Request: prompt_tokens={prompt_tokens}, max_tokens={max_tokens}")

    if stream:
        return StreamingResponse(
            generate_stream(inputs, max_tokens, temperature),
            media_type="text/event-stream",
        )
    else:
        result = generate_sync(inputs, max_tokens, temperature)
        return result


def generate_sync(inputs, max_tokens, temperature):
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


async def generate_stream(inputs, max_tokens, temperature):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    for text in streamer:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": text} if text else {},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join()


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\models\Qwen3.5-2B")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    load_model(args.model, args.device)
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
