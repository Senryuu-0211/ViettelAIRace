import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pipeline.langgraph_extractor.graph import create_extractor_chain, parse_llm_json
from pipeline.langgraph_extractor.prompts.thuoc_prompt import THUOC_SYSTEM_PROMPT, THUOC_USER_TEMPLATE

text = """
Tiền sử bệnh: Thuốc trước khi nhập viện
- metoprolol 25mg po bid
"""

chain = create_extractor_chain(THUOC_SYSTEM_PROMPT, THUOC_USER_TEMPLATE)
print("Invoking LLM...")
raw = chain.invoke({"input_text": text})
print("--- RAW LLM OUTPUT ---")
print(raw)
print("--- PARSED JSON ---")
print(parse_llm_json(raw))
