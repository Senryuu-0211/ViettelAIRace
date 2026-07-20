# -*- coding: utf-8 -*-
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pipeline.config import OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE
from pipeline.langgraph_extractor.prompts.thuoc_prompt import THUOC_SYSTEM_PROMPT, THUOC_USER_TEMPLATE

input_path = os.path.join(os.path.dirname(__file__), "input", "1.txt")
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=TEMPERATURE, num_ctx=8192)
prompt = ChatPromptTemplate.from_messages([
    ("system", THUOC_SYSTEM_PROMPT),
    ("human", THUOC_USER_TEMPLATE),
])

extract_content = lambda msg: msg.content if hasattr(msg, "content") else str(msg)
chain = prompt | llm | extract_content

print(f"Invoking THUOC chain (len={len(text)} chars)...")
t0 = time.time()
try:
    raw = chain.invoke({"input_text": text})
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Content len: {len(raw)}")
    print(f"Content: {raw[:300]}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"Error after {elapsed:.1f}s: {e}")
