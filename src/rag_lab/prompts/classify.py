from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# ── Output schema ─────────────────────────────────────────────────────────────

QueryType = Literal["factual", "comparative", "summarization", "out_of_scope"]


class ClassifyOutput(BaseModel):
    """Structured output for the query classification node.

    query_type: The category the query falls into.
        - factual:        asks for a specific fact, date, definition, or value
        - comparative:    asks to compare or contrast two or more things
        - summarization:  asks for an overview or summary of a topic
        - out_of_scope:   cannot be answered from document content
                          (e.g. real-time data, personal opinions, unrelated topics)

    reasoning: One sentence explaining why this category was chosen.
               Captured for LangSmith traces and UI display.
    """

    query_type: QueryType
    reasoning: str


# ── Prompt template ───────────────────────────────────────────────────────────

classify_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a query classifier for a document Q&A system. "
                "Your job is to categorize the user's query into exactly one of: "
                "factual, comparative, summarization, or out_of_scope. "
                "Base your decision only on the query itself, not on any documents."
            ),
        ),
        ("human", "{query}"),
    ]
)
