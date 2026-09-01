"""Embedding text construction and chunking for Vector Search indexing.

Used by the Enrichment_Pipeline to build the `embedding_text` field
(Requirement 4.2) and to split overly long text into overlapping chunks
before indexing (Requirement 4.3).
"""

from typing import List


def build_embedding_text(job_title: str, job_description: str, required_skills_text: str) -> str:
    """Build the embedding text for a Job_Listing.

    Concatenates `job_title`, `job_description`, and `required_skills_text`
    separated by single spaces, matching the design's
    `job_title || ' ' || job_description || ' ' || required_skills_text` pattern.

    Args:
        job_title: The job title of the listing.
        job_description: The full job description text.
        required_skills_text: The comma-joined required skills text.

    Returns:
        The concatenated embedding text.
    """
    return f"{job_title} {job_description} {required_skills_text}"


def chunk_text(text: str, max_tokens: int = 512, overlap_tokens: int = 64) -> List[str]:
    """Split long text into overlapping chunks for embedding.

    Uses whitespace-splitting as an approximate token-count proxy (no
    heavy NLP tokenizer dependency is required by the design). Chunks are
    built by sliding a window of `max_tokens` words with a step of
    `max_tokens - overlap_tokens` words, so consecutive chunks overlap by
    `overlap_tokens` words.

    Args:
        text: The text to chunk.
        max_tokens: The maximum number of words (token proxy) per chunk.
        overlap_tokens: The number of words of overlap between consecutive
            chunks.

    Returns:
        A list of chunk strings. If `text` is empty, returns an empty list.
        If the text contains at most `max_tokens` words, returns a single
        element list containing the original text unchanged.

    Raises:
        ValueError: If `max_tokens` is not a positive integer, or if
            `overlap_tokens` is negative or `overlap_tokens >= max_tokens`.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must not be negative")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    if text == "":
        return []

    words = text.split()

    if len(words) <= max_tokens:
        return [text]

    step = max_tokens - overlap_tokens
    chunks: List[str] = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + max_tokens, n)
        chunks.append(" ".join(words[start:end]))
        if end == n:
            break
        start += step

    return chunks
