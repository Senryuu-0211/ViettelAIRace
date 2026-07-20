import json
import re
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Import Prompts
from pipeline.langgraph_extractor.prompts.thuoc_prompt import THUOC_SYSTEM_PROMPT, THUOC_USER_TEMPLATE
from pipeline.langgraph_extractor.prompts.chandoan_prompt import CHANDOAN_SYSTEM_PROMPT, CHANDOAN_USER_TEMPLATE
from pipeline.langgraph_extractor.prompts.trieuchung_prompt import TRIEUCHUNG_SYSTEM_PROMPT, TRIEUCHUNG_USER_TEMPLATE
from pipeline.langgraph_extractor.prompts.ten_xn_prompt import TENXN_SYSTEM_PROMPT, TENXN_USER_TEMPLATE
from pipeline.langgraph_extractor.prompts.kq_xn_prompt import KQXN_SYSTEM_PROMPT, KQXN_USER_TEMPLATE

from pipeline.config import OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE

class ExtractionState(TypedDict):
    input_text: str
    results: Annotated[list[dict], operator.add]

def create_extractor_chain(system_prompt: str, user_template: str, num_ctx: int = None):
    kwargs = dict(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=TEMPERATURE)
    if num_ctx is not None:
        kwargs["num_ctx"] = num_ctx
    llm = ChatOllama(**kwargs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template),
    ])
    return prompt | llm | (lambda msg: msg.content if hasattr(msg, "content") else str(msg))

# Helper parse JSON (Thừa kế sự robust từ file cũ của bạn)
def parse_llm_json(raw: str) -> list[dict]:
    raw = raw.strip()
    for prefix in ("```json", "```"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    for suffix in ("```",):
        if raw.endswith(suffix):
            raw = raw[:-len(suffix)]
    raw = raw.strip()

    result = []
    decoder = json.JSONDecoder()
    pos = 0

    while pos < len(raw):
        try:
            obj, new_pos = decoder.raw_decode(raw, pos)
            items = obj if isinstance(obj, list) else [obj] if isinstance(obj, dict) else []
            for item in items:
                if isinstance(item, dict):
                    result.append(item)
            pos = new_pos
        except json.JSONDecodeError:
            pos += 1

    if not result:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            result.append(item)
            except json.JSONDecodeError:
                pass
    return result

# --- Nodes ---
def node_extract_thuoc(state: ExtractionState):
    chain = create_extractor_chain(THUOC_SYSTEM_PROMPT, THUOC_USER_TEMPLATE)
    raw = chain.invoke({"input_text": state["input_text"]})
    return {"results": parse_llm_json(raw)}

def node_extract_chandoan(state: ExtractionState):
    chain = create_extractor_chain(CHANDOAN_SYSTEM_PROMPT, CHANDOAN_USER_TEMPLATE)
    raw = chain.invoke({"input_text": state["input_text"]})
    return {"results": parse_llm_json(raw)}

def node_extract_trieuchung(state: ExtractionState):
    chain = create_extractor_chain(TRIEUCHUNG_SYSTEM_PROMPT, TRIEUCHUNG_USER_TEMPLATE)
    raw = chain.invoke({"input_text": state["input_text"]})
    return {"results": parse_llm_json(raw)}

def node_extract_ten_xn(state: ExtractionState):
    chain = create_extractor_chain(TENXN_SYSTEM_PROMPT, TENXN_USER_TEMPLATE)
    raw = chain.invoke({"input_text": state["input_text"]})
    return {"results": parse_llm_json(raw)}

def node_extract_kq_xn(state: ExtractionState):
    chain = create_extractor_chain(KQXN_SYSTEM_PROMPT, KQXN_USER_TEMPLATE)
    raw = chain.invoke({"input_text": state["input_text"]})
    return {"results": parse_llm_json(raw)}

# --- Graph Builder ---
def build_extraction_graph():
    workflow = StateGraph(ExtractionState)

    workflow.add_node("extract_thuoc", node_extract_thuoc)
    workflow.add_node("extract_chandoan", node_extract_chandoan)
    workflow.add_node("extract_trieuchung", node_extract_trieuchung)
    workflow.add_node("extract_ten_xn", node_extract_ten_xn)
    workflow.add_node("extract_kq_xn", node_extract_kq_xn)

    # Từ START bắn ra cả 5 nodes cùng lúc
    workflow.add_edge(START, "extract_thuoc")
    workflow.add_edge(START, "extract_chandoan")
    workflow.add_edge(START, "extract_trieuchung")
    workflow.add_edge(START, "extract_ten_xn")
    workflow.add_edge(START, "extract_kq_xn")

    # Từ 5 nodes hội tụ về END (LangGraph tự merge vào list `results` nhờ operator.add)
    workflow.add_edge("extract_thuoc", END)
    workflow.add_edge("extract_chandoan", END)
    workflow.add_edge("extract_trieuchung", END)
    workflow.add_edge("extract_ten_xn", END)
    workflow.add_edge("extract_kq_xn", END)

    return workflow.compile()

