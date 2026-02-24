from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# ── Output schema ─────────────────────────────────────────────────────────────


class AnswerOutput(BaseModel):
    """Structured output for the answer generation node.

    answer: The generated answer text. If the chunks do not contain enough
            information, the answer should say so explicitly rather than
            hallucinating. The grounding node will catch hallucinations, but
            prompting for honesty reduces them upstream.

    cited_chunk_indices: Zero-based indices into the retrieved_chunks list
                         identifying which chunks were actually used to produce
                         the answer. The UI uses these to highlight cited chunks
                         and the grounding node uses them to verify support.
                         An empty list means no chunks were cited (e.g. the LLM
                         determined the context was insufficient).
    """

    answer: str
    cited_chunk_indices: list[int]


# ── Prompt template ───────────────────────────────────────────────────────────

# {query_type} is passed so the LLM can tailor its answer style:
#   - factual      → direct, specific answer
#   - comparative  → structured comparison
#   - summarization → coherent overview
#   - out_of_scope → polite refusal (though this is also handled by the edge
#                    conditions in Phase 2; in Phase 1 all queries reach here)

# The instruction to cite by index (not by quoting the chunk) is intentional:
# it keeps the answer concise and forces explicit attribution that the grounding
# node can verify programmatically.

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a precise Q&A assistant. Answer the user's query using "
                "only the information in the provided document chunks. "
                "Do not use prior knowledge or make up information. "
                "If the chunks do not contain enough information to answer, "
                "say so clearly instead of guessing. "
                "Cite the chunks you use by their index numbers (e.g. [0], [2])."
            ),
        ),
        (
            "human",
            "Query type: {query_type}\n\nQuery: {query}\n\nDocument chunks:\n{context}",
        ),
    ]
)
