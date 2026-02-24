from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# ── Output schema ─────────────────────────────────────────────────────────────


class GradeOutput(BaseModel):
    """Structured output for the context grading node.

    grade: Whether the retrieved chunks contain enough information to answer
           the query.
        - sufficient:    the chunks, taken together, provide enough information
                         to produce a well-grounded answer
        - insufficient:  the chunks are missing key information, are off-topic,
                         or are too vague to answer the query

    reasoning: One or two sentences explaining the grade.
               Used in LangSmith traces and to inform the rewrite_query node
               about *why* the context fell short.
    """

    grade: Literal["sufficient", "insufficient"]
    reasoning: str


# ── Prompt template ───────────────────────────────────────────────────────────

# The {context} variable is a pre-formatted string assembled by the node:
#   "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(chunks))
# Numbered chunk labels let the LLM refer to specific chunks in its reasoning,
# and the same indices are reused by the answer node for citation.

grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a context grader for a document Q&A system. "
                "Given a query and a set of retrieved document chunks, decide whether "
                "the chunks contain sufficient information to answer the query. "
                "Be strict: if the chunks are missing key facts, are off-topic, or "
                "only partially address the query, grade them as insufficient."
            ),
        ),
        (
            "human",
            "Query: {query}\n\nRetrieved chunks:\n{context}",
        ),
    ]
)
