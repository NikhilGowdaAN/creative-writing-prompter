"""Tests for Creative Writing Prompter core engine."""

import pytest
from app.core import (
    Corpus,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    tokenize,
)


# Milestone 1: Prepare a text corpus of short stories.
def test_milestone_1():
    corpus = milestone_1()
    assert isinstance(corpus, Corpus)
    assert len(corpus.tokens) > 0
    assert corpus.vocab_size > 0
    assert "fantasy" in corpus.genres
    assert "scifi" in corpus.genres
    assert "mystery" in corpus.genres
    assert "horror" in corpus.genres
    assert "adventure" in corpus.genres

    # Check tokenization properties
    assert all(isinstance(token, str) for token in corpus.tokens)
    assert all(token == token.lower() for token in corpus.tokens)

    # Check specific tokens from the stories
    assert "dragon" in corpus.vocabulary
    assert "starship" in corpus.vocabulary
    assert "investigation" in corpus.vocabulary or "inspector" in corpus.vocabulary

    # Test filtering by genre
    fantasy_corpus = milestone_1(genre="fantasy")
    assert fantasy_corpus.genres == ["fantasy"]
    assert "dragon" in fantasy_corpus.vocabulary
    assert "starship" not in fantasy_corpus.vocabulary


def test_tokenize_helper():
    text = "Don't stop, the clock struck 12! It's ticking..."
    words = tokenize(text, lowercase=True, strip_punct=True)
    assert "don't" in words
    assert "stop" in words
    assert "it's" in words
    assert "," not in words
    assert "!" not in words

    tokens_with_punct = tokenize(text, lowercase=True, strip_punct=False)
    assert "," in tokens_with_punct
    assert "!" in tokens_with_punct


# Milestone 2: Implement n-gram counting and maximum-likelihood probabilities.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Add add-k smoothing to handle unseen n-grams.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Develop a function to generate text by sampling from the model.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Create a CLI interface for users to input text and get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
