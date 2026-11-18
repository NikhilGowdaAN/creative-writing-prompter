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
def test_milestone_3():
    # Setup a small deterministic model:
    # vocab = {"cat", "dog", "mouse"}, |V| = 3
    # "the" is followed by: "cat" twice, "dog" once. "mouse" never follows "the".
    tokens = ["the", "cat", "the", "cat", "the", "dog", "a", "mouse"]
    model = NGramModel(n=2)
    model.train(tokens)
    v_size = len(model.vocabulary)
    assert v_size == 5  # {"the", "cat", "dog", "a", "mouse"}

    # Under MLE, unseen bigram ("the", "mouse") has probability 0.0
    assert model.mle_prob("the", "mouse") == 0.0

    # With Laplace smoothing (k=1.0):
    # C("the", "mouse") = 0, C("the") = 3, |V| = 5
    # P("mouse" | "the") = (0 + 1) / (3 + 1 * 5) = 1 / 8 = 0.125
    smoothed_p = model.smoothed_prob("the", "mouse", k=1.0)
    assert smoothed_p == 0.125

    # Check seen follower: C("the", "cat") = 2
    # P("cat" | "the") = (2 + 1) / (3 + 5) = 3 / 8 = 0.375
    assert model.smoothed_prob("the", "cat", k=1.0) == 0.375

    # Full distribution over vocabulary sums to 1.0
    dist = model.get_distribution("the", k=1.0)
    assert len(dist) == v_size
    assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0

    # Test with fractional k (e.g. Lidstone smoothing k=0.5)
    dist_k05 = model.get_distribution("the", k=0.5)
    assert pytest.approx(sum(dist_k05.values()), rel=1e-5) == 1.0

    # Test completely unseen context
    # C("mysterious") = 0
    # Every word in V should get 1 / |V| = 1 / 5 = 0.2
    unseen_dist = model.get_distribution("mysterious", k=1.0)
    for word, prob in unseen_dist.items():
        assert prob == 1.0 / v_size
    assert pytest.approx(sum(unseen_dist.values()), rel=1e-5) == 1.0

    # Test milestone_3 helper on full corpus
    full_model = milestone_3(k=0.5)
    assert full_model.smoothed_prob("ancient", "dragon", k=0.5) > 0.0
    # An unseen combination in corpus
    assert full_model.smoothed_prob("ancient", "cygnus", k=0.5) > 0.0

    # Perplexity is finite
    perp = full_model.perplexity(["ancient", "dragon", "flew", "away"], k=0.5)
    assert perp > 0.0 and perp != float("inf")


# Milestone 4: Develop a function to generate text by sampling from the model.
def test_milestone_4():
    tokens = ["the", "ancient", "grimoire", "the", "ancient", "grimoire", "the", "ancient", "temple"]
    model = NGramModel(n=2)
    model.train(tokens)

    # 1. Test greedy sampling (temperature = 0.0)
    # After "ancient", "grimoire" occurs twice, "temple" occurs once.
    # Greedy must deterministically pick "grimoire"
    greedy_word = model.sample_next_word("ancient", k=1.0, temperature=0.0)
    assert greedy_word == "grimoire"

    # 2. Test suggest_next_words
    suggestions = model.suggest_next_words("ancient", num_suggestions=2, k=1.0)
    assert len(suggestions) == 2
    assert suggestions[0][0] == "grimoire"
    assert suggestions[1][0] == "temple"
    assert suggestions[0][1] > suggestions[1][1]

    # 3. Test reproducible sampling with seed
    word1 = model.sample_next_word("ancient", k=1.0, temperature=1.0, seed=123)
    word2 = model.sample_next_word("ancient", k=1.0, temperature=1.0, seed=123)
    assert word1 == word2

    # 4. Test text generation
    generated = model.generate(prompt="the ancient", max_words=4, k=1.0, temperature=0.0)
    words = generated.split()
    assert words[0:2] == ["the", "ancient"]
    assert len(words) == 6  # 2 prompt words + 4 generated words

    # 5. Test milestone_4 function with full corpus
    text = milestone_4(prompt="The ancient", max_words=10, seed=42)
    assert isinstance(text, str)
    assert text.lower().startswith("the ancient")
    gen_words = text.split()
    assert len(gen_words) == 12  # 2 prompt + 10 generated words
    assert all(isinstance(w, str) for w in gen_words)

    # 6. Test trigram text generation
    trigram_text = milestone_4(prompt="The ancient archives", max_words=5, n=3, seed=42)
    assert trigram_text.lower().startswith("the ancient archives")
    assert len(trigram_text.split()) == 8


# Milestone 5: Create a CLI interface for users to input text and get suggestions.
def test_milestone_5(capsys):
    # 1. Test suggest command via milestone_5
    milestone_5(["suggest", "the ancient", "--top", "3", "--genre", "fantasy"])
    out = capsys.readouterr().out
    assert "NEXT-WORD SUGGESTIONS" in out
    assert "the ancient" in out
    assert "[1]" in out

    # 2. Test generate command
    milestone_5(["generate", "Commander Varek", "--length", "6", "--genre", "scifi", "--seed", "42"])
    out = capsys.readouterr().out
    assert "STORY INSPIRATION GENERATOR" in out
    assert "commandervarek" in out.lower() or "commander varek" in out.lower()

    # 3. Test corpus stats command
    milestone_5(["corpus"])
    out = capsys.readouterr().out
    assert "GENRE CORPUS DETAILS" in out
    assert "fantasy" in out
    assert "scifi" in out

    # 4. Test default milestone_5 invocation
    milestone_5()
    out = capsys.readouterr().out
    assert "NEXT-WORD SUGGESTIONS" in out

    # 5. Test save flag with store
    milestone_5(["suggest", "the dragon", "--top", "3", "--save"])
    out = capsys.readouterr().out
    assert "Saved suggestions to session history" in out


def test_cli_interactive(monkeypatch, capsys):
    # Simulate user typing "1" (select first suggestion) then "/quit"
    inputs = iter(["1", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    milestone_5(["interactive", "--genre", "fantasy"])
    out = capsys.readouterr().out
    assert "CREATIVE WRITING ASSISTANT" in out
    assert "Ending session" in out
    assert "Keep writing!" in out
