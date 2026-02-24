from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# ── Output schema ─────────────────────────────────────────────────────────────


class GroundingOutput(BaseModel):
    """Structured output for the grounding verification node.

    grounding_score: A float between 0.0 and 1.0 representing how well the
                     answer is supported by the cited chunks.
                       1.0 = every claim in the answer is directly supported
                       0.5 = answer is partially supported, some claims unverified
                       0.0 = answer is not supported by the chunks at all

    reasoning: One or two sentences explaining the score.
               Surfaced in the UI and LangSmith traces.

    Note: pass/fail is NOT part of this schema. The node computes it by
    comparing grounding_score against settings.grounding_threshold. Keeping
    the threshold logic in Python (not the LLM) means we can adjust the
    threshold without changing the prompt.
    """

    grounding_score: float
    reasoning: str


# ── Prompt template ───────────────────────────────────────────────────────────

# {cited_context} is a pre-formatted string containing only the chunks that
# were cited by the answer node (not all retrieved chunks). This focuses the
# LLM's attention on the specific evidence the answer claimed to use.
# Assembled by the node as:
#   "\n\n".join(
#       f"[{i}] {retrieved_chunks[i].page_content}"
#       for i in cited_chunk_indices
#   )

grounding_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a grounding verifier for a document Q&A system. "
                "Given a query, a generated answer, and the document chunks the "
                "answer claims to be based on, score how well the answer is "
                "supported by those chunks. "
                "A score of 1.0 means every claim in the answer is directly "
                "traceable to the provided chunks. "
                "A score of 0.0 means the answer contains claims not found in "
                "the chunks at all. "
                "Be strict: unsupported claims should significantly lower the score."
            ),
        ),
        (
            "human",
            "Query: {query}\n\nAnswer: {answer}\n\nCited chunks:\n{cited_context}",
        ),
    ]
)
