"""Tests for Creative Writing Prompter core engine."""

import pytest
from app.core import (
    Corpus,
    NGramModel,
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
def test_milestone_2():
    # Test on a small deterministic sequence first
    sample_tokens = ["the", "ancient", "grimoire", "the", "ancient", "temple", "the", "dark", "citadel"]
    model = milestone_2(tokens=sample_tokens, n=2)
    assert model.n == 2
    assert model.total_tokens == len(sample_tokens)

    # Context 'ancient' was followed by 'grimoire' (1) and 'temple' (1)
    assert model.count("ancient", "grimoire") == 1
    assert model.count("ancient", "temple") == 1
    assert model.context_count("ancient") == 2
    assert model.mle_prob("ancient", "grimoire") == 0.5
    assert model.mle_prob("ancient", "temple") == 0.5
    assert model.mle_prob("ancient", "citadel") == 0.0

    # Probabilities of observed followers sum to 1.0
    candidates = model.mle_candidates("ancient")
    assert len(candidates) == 2
    total_prob = sum(p for _, p in candidates)
    assert pytest.approx(total_prob, rel=1e-5) == 1.0

    # Test on full corpus
    corpus_model = milestone_2(n=2)
    assert corpus_model.total_tokens > 0
    # In fantasy story, 'ancient' appears
    assert corpus_model.context_count("ancient") >= 1
    ancient_followers = corpus_model.mle_candidates("ancient")
    assert len(ancient_followers) >= 1
    assert pytest.approx(sum(p for _, p in ancient_followers), rel=1e-5) == 1.0

    # Trigram model (n=3)
    trigram_model = milestone_2(tokens=sample_tokens, n=3)
    assert trigram_model.count(("the", "ancient"), "grimoire") == 1
    assert trigram_model.count(("the", "ancient"), "temple") == 1
    assert trigram_model.mle_prob(("the", "ancient"), "grimoire") == 0.5


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
