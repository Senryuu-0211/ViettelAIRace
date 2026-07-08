from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import ChatOllama

from pipeline.config import OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE
from pipeline.prompts.assertion_batch_template import ASSERTION_BATCH_SYSTEM_PROMPT, ASSERTION_BATCH_USER_TEMPLATE

_chain: Optional[RunnableSerializable] = None


def get_assertion_chain() -> RunnableSerializable:
    global _chain
    if _chain is None:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", ASSERTION_BATCH_SYSTEM_PROMPT),
            ("human", ASSERTION_BATCH_USER_TEMPLATE),
        ])
        _chain = prompt | llm | StrOutputParser()
    return _chain
