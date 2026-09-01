"""Property-based tests for cover letter word count validation and empty
profile draft rejection.

Property 22: Cover Letter Word Count Validation
Validates: Requirements 8.3

Property 23: Empty Profile Draft Rejection
Validates: Requirements 8.8

The real logic lives inside the `draft_application` UC Function's SQL/LLM
prompt body (see `notebooks/06_create_uc_functions.py`), which must produce
a 200-500 word cover letter and must refuse to draft when the User_Profile
has no skills and no job title history. This test models both invariants
as pure-Python helpers.
"""

from hypothesis import given, strategies as st


def is_valid_cover_letter_length(text: str) -> bool:
    """Return True iff `text`'s whitespace-based word count is 200-500.

    Mirrors Requirement 8.3: the Application_Drafter must produce a cover
    letter of 200 to 500 words inclusive.
    """
    word_count = len(text.split())
    return 200 <= word_count <= 500


def should_reject_draft(user_skills: list, user_job_titles: list) -> bool:
    """Return True iff the draft request should be rejected.

    Mirrors the UC Function's `if not user_skills and not user_job_titles`
    check (Requirement 8.8): the Application_Drafter must refuse to
    generate a cover letter when the User_Profile has 0 skills entries and
    0 job title history entries.

    Args:
        user_skills: the User_Profile skills list (may be None or empty).
        user_job_titles: the User_Profile job title history list (may be
            None or empty).

    Returns:
        True if both lists are falsy (None or empty), False otherwise.
    """
    return not user_skills and not user_job_titles


# --- Property 22: Cover Letter Word Count Validation ------------------------

_word = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=10
)


@given(word_count=st.integers(min_value=0, max_value=800), words=st.data())
def test_word_count_classification_matches_actual_count(word_count, words):
    """**Validates: Requirements 8.3**

    For a text built as `word_count` space-joined words, the helper
    classifies it as valid iff `word_count` truly lies within [200, 500].
    """
    generated_words = words.draw(st.lists(_word, min_size=word_count, max_size=word_count))
    text = " ".join(generated_words)

    result = is_valid_cover_letter_length(text)
    expected = 200 <= word_count <= 500

    assert result == expected


@given(word_count=st.integers(min_value=200, max_value=500), words=st.data())
def test_in_range_word_counts_are_valid(word_count, words):
    """**Validates: Requirements 8.3**

    Any text with a word count in [200, 500] is classified as valid.
    """
    generated_words = words.draw(st.lists(_word, min_size=word_count, max_size=word_count))
    text = " ".join(generated_words)

    assert is_valid_cover_letter_length(text) is True


@given(
    word_count=st.one_of(
        st.integers(min_value=0, max_value=199),
        st.integers(min_value=501, max_value=1000),
    ),
    words=st.data(),
)
def test_out_of_range_word_counts_are_invalid(word_count, words):
    """**Validates: Requirements 8.3**

    Any text with a word count outside [200, 500] is classified as
    invalid.
    """
    generated_words = words.draw(st.lists(_word, min_size=word_count, max_size=word_count))
    text = " ".join(generated_words)

    assert is_valid_cover_letter_length(text) is False


# --- Property 23: Empty Profile Draft Rejection ------------------------------

_optional_list = st.one_of(
    st.none(),
    st.just([]),
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10),
)


@given(user_skills=_optional_list, user_job_titles=_optional_list)
def test_rejection_iff_both_lists_empty_or_none(user_skills, user_job_titles):
    """**Validates: Requirements 8.8**

    Drafting is rejected if and only if both `user_skills` and
    `user_job_titles` are empty or None -- if either list has at least one
    entry, drafting must proceed.
    """
    result = should_reject_draft(user_skills, user_job_titles)
    expected = not user_skills and not user_job_titles

    assert result == expected


@given(
    user_skills=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10),
    user_job_titles=_optional_list,
)
def test_non_empty_skills_prevents_rejection(user_skills, user_job_titles):
    """**Validates: Requirements 8.8**

    If the skills list has at least one entry, the draft is never
    rejected, regardless of the job title history.
    """
    assert should_reject_draft(user_skills, user_job_titles) is False


@given(
    user_skills=_optional_list,
    user_job_titles=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10),
)
def test_non_empty_job_titles_prevents_rejection(user_skills, user_job_titles):
    """**Validates: Requirements 8.8**

    If the job title history has at least one entry, the draft is never
    rejected, regardless of the skills list.
    """
    assert should_reject_draft(user_skills, user_job_titles) is False


def test_both_empty_lists_are_rejected():
    """**Validates: Requirements 8.8**

    Explicit example: 0 skills entries and 0 job title history entries
    must be rejected.
    """
    assert should_reject_draft([], []) is True


def test_both_none_are_rejected():
    """**Validates: Requirements 8.8**

    Explicit example: None for both fields (field absent entirely) must
    also be rejected.
    """
    assert should_reject_draft(None, None) is True
