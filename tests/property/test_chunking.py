"""Property-based tests for text chunking.

Property 12: Text Chunking Correctness
Validates: Requirements 4.3
"""

from hypothesis import given, strategies as st

from src.pipelines.embedding import chunk_text

# Base word alphabet kept simple (lowercase letters) - uniqueness is enforced
# by appending the word's position, so we can track exactly which original
# word ended up in which chunk without relying on hypothesis to generate
# distinct strings itself.
_word_strategy = st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=5)


def _make_words(raw_words):
    """Turn raw random words into a list of position-unique tokens."""
    return [f"{w}_{i}" for i, w in enumerate(raw_words)]


@given(
    raw_words=st.lists(_word_strategy, min_size=0, max_size=300),
    max_tokens=st.integers(min_value=1, max_value=50),
    overlap_tokens=st.integers(min_value=0, max_value=49),
)
def test_chunk_text_properties(raw_words, max_tokens, overlap_tokens):
    """**Validates: Requirements 4.3**

    For arbitrary text (as space-joined, position-unique words) and valid
    max_tokens/overlap_tokens combinations, chunk_text always:
    (a) returns [] for empty text
    (b) returns a single chunk equal to the input when word count <= max_tokens
    (c) for longer text, every chunk has word count <= max_tokens
    (d) chunks cover every original word position with no drops
    """
    if overlap_tokens >= max_tokens:
        # Not a valid combination for this property; validity is covered
        # separately in test_chunk_text_invalid_arguments_raise.
        return

    words = _make_words(raw_words)
    text = " ".join(words)
    n = len(words)

    chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    if n == 0:
        # (a) empty text -> []
        assert chunks == []
        return

    if n <= max_tokens:
        # (b) short text -> single chunk equal to the input
        assert chunks == [text]
        return

    # (c) every chunk respects the max_tokens cap, and chunk contents are a
    # contiguous run of original word positions.
    tag_to_index = {tag: i for i, tag in enumerate(words)}
    covered = set()
    chunk_index_runs = []
    for chunk in chunks:
        chunk_words = chunk.split()
        assert len(chunk_words) <= max_tokens

        indices = [tag_to_index[w] for w in chunk_words]
        # Each chunk must be a contiguous slice of the original word order.
        assert indices == list(range(indices[0], indices[-1] + 1))

        chunk_index_runs.append(indices)
        covered.update(indices)

    # (d) no words are dropped: every original position is covered by at
    # least one chunk.
    assert covered == set(range(n))

    # The first chunk must start at the very first word and the last chunk
    # must end at the very last word (otherwise the boundaries would leave
    # a gap at the start/end rather than in the middle).
    assert chunk_index_runs[0][0] == 0
    assert chunk_index_runs[-1][-1] == n - 1


@given(
    max_tokens=st.integers(min_value=-10, max_value=50),
    overlap_tokens=st.integers(min_value=-10, max_value=60),
)
def test_chunk_text_invalid_arguments_raise(max_tokens, overlap_tokens):
    """**Validates: Requirements 4.3**

    (e) chunk_text raises ValueError for invalid max_tokens/overlap_tokens
    combinations: max_tokens <= 0, overlap_tokens < 0, or
    overlap_tokens >= max_tokens.
    """
    is_invalid = max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens
    if not is_invalid:
        return

    try:
        chunk_text("some sample text here", max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"Expected ValueError for max_tokens={max_tokens}, overlap_tokens={overlap_tokens}"
        )
