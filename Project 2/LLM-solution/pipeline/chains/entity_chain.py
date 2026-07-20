from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import ChatOllama

from pipeline.config import OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE
from pipeline.prompts.entity_template import ENTITY_SYSTEM_PROMPT, ENTITY_USER_TEMPLATE

_chain: Optional[RunnableSerializable] = None


def get_entity_chain() -> RunnableSerializable:
    global _chain
    if _chain is None:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", ENTITY_SYSTEM_PROMPT),
            ("human", ENTITY_USER_TEMPLATE),
        ])
        _chain = prompt | llm | StrOutputParser()
    return _chain
