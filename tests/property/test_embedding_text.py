"""Property-based tests for embedding text construction.

Property 11: Embedding Text Construction
Validates: Requirements 4.2
"""

from hypothesis import given, strategies as st

from src.pipelines.embedding import build_embedding_text


@given(
    job_title=st.text(),
    job_description=st.text(),
    required_skills_text=st.text(),
)
def test_build_embedding_text_exact_concatenation(job_title, job_description, required_skills_text):
    """**Validates: Requirements 4.2**

    For arbitrary strings, build_embedding_text always returns the exact
    expected concatenation "{job_title} {job_description} {required_skills_text}".
    """
    result = build_embedding_text(job_title, job_description, required_skills_text)
    assert result == f"{job_title} {job_description} {required_skills_text}"


@given(
    job_title=st.text(min_size=1),
    job_description=st.text(min_size=1),
    required_skills_text=st.text(min_size=1),
)
def test_build_embedding_text_contains_all_inputs(job_title, job_description, required_skills_text):
    """**Validates: Requirements 4.2**

    When each input is non-empty, the result must contain all three input
    substrings.
    """
    result = build_embedding_text(job_title, job_description, required_skills_text)
    assert job_title in result
    assert job_description in result
    assert required_skills_text in result
