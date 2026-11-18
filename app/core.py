"""Core next-word prediction engine and creative writing prompter."""

from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


# Default fallback stories embedded directly so the engine is self-contained.
BUILTIN_STORIES: dict[str, str] = {
    "fantasy": (
        "The ancient archives of Eldoria stood atop the whispering cliffs, where the wind carried "
        "fragments of forgotten incantations. Eldrin, a scholar-mage of the third circle, traced "
        "his fingers along the weathered spine of a leather-bound grimoire. Dust rose in shimmering "
        "motes, dancing in the amber moonlight that filtered through the arched stained-glass windows. "
        "Outside the citadel, the forest of Silverwood hummed with restless energy. Mythical creatures "
        "stirred in the emerald shadows. Rumors spoke of a dormant dragon slumbering beneath the obsidian "
        "crags of Mount Forlorn, guarding a relic that could alter the flow of magic across the kingdom. "
        "Eldrin knew that if the seal were breached, darkness would sweep over the mortal realms. "
        "He adjusted his azure robe, gripped his carved ash staff, and chanted the words of revealing. "
        "A faint turquoise luminescence pulsed along the floor tiles, uncovering a hidden trapdoor "
        "marked with the crest of the sun-forged kings. Taking a steadying breath, the mage descended "
        "into the winding subterranean vaults, where forgotten secrets and perilous enchantments "
        "awaited those brave enough to seek the truth."
    ),
    "scifi": (
        "The hum of the ion propulsion thrusters resonated through the titanium hull of the starship "
        "Vanguard. Commander Varek peered through the observation port at the swirling violet accretion "
        "disk of the Cygnus singularity. Deep space was never truly silent; the ship's neural core "
        "emitted a soft synthetic purr as it mapped quantum anomalies across the spiral arm. "
        "An amber alert flashed upon the holographic navigation console. A derelict transmission was "
        "drifting from an uncharted asteroid cluster near the galactic rim. The signal was encoded in "
        "binary waveforms that defied standard planetary protocols. Beside him, unit Seven-Echo, a "
        "synthetic navigator with glowing cobalt optic sensors, calibrated the long-range sensor telemetry. "
        "'Captain, the energy signature does not match any registered human colony,' Seven-Echo reported, "
        "its voice calm and modulated. Varek tapped the command interface, rerouting auxiliary reactor "
        "power to the forward deflector shields. The void beyond held answers to questions humanity had "
        "pondered for centuries, but every deep space voyage taught one unforgiving law: the cosmos "
        "belonged to those prepared for the unknown."
    ),
    "mystery": (
        "Rain lashed against the frosted glass of Inspector Silas Vance's dimly lit office on Baker Row. "
        "Street lamps cast elongated silhouettes across the wet cobblestones below, where fog rolled in "
        "heavy waves off the river. On Silas's cluttered oak desk lay a torn telegram and an antique silver "
        "pocket watch stopped precisely at midnight. "
        "The victim was Lord Harrington, an eccentric collector found slumped in his locked study at "
        "Blackwood Manor. There were no signs of forced entry, no footprints in the mud beneath the "
        "garden terrace, and only a faint aroma of bitter almonds lingering in the air. A single brass "
        "key was clenched in Harrington's cold fingers, but none of the cabinets in the house matched "
        "its peculiar design. Silas pulled up the collar of his trench coat and struck a match to ignite "
        "his pipe. Every crime scene possessed a rhythm, an invisible thread connecting motive to "
        "opportunity. Harrington had harbored enemies in political circles and antiquities syndicates alike. "
        "Vance picked up his leather notebook and stepped out into the stormy night; in this city, shadows "
        "rarely guarded innocent secrets for long."
    ),
    "horror": (
        "The iron gates of Ravenscroft Manor groaned under the force of the nocturnal tempest. Thunder "
        "rattled the leaded glass windows, casting brief skeletal flashes upon the rotting tapestries "
        "that lined the grand hallway. Clara had arrived at sundown, summoned by an unsigned parchment "
        "warning her never to return to her ancestral home. "
        "A pervasive cold permeated the drafty corridors, smelling of damp earth and decayed wood. In the "
        "parlor, an antique grandfather clock chimed thirteen discordant tolls before the pendulum ceased "
        "swinging. Clara raised her brass candelabra, watching the flickering flame quiver as though "
        "disturbed by an unseen breath. Behind the wallpaper, faint scratching noises echoed, followed "
        "by a soft, mournful weeping that seemed to emanate directly from the floorboards beneath her feet. "
        "She knew she should flee into the howling squall, but the heavy oak doors had slammed shut with "
        "a sickening thud, their iron bolts sliding forward without human intervention. In the suffocating "
        "dark, silence was not empty; it was patient, hungry, and watching."
    ),
    "adventure": (
        "The jungle canopy choked out the midday sun, casting the tangled river valley in perpetual "
        "twilight. Maya wiped beads of sweat from her brow as she slashed through the dense emerald "
        "foliage with her machete. According to the sun-faded parchment in her satchel, the lost temple "
        "of the Obsidian Serpent lay somewhere past the roaring cataracts of the Rio Verde. "
        "Her compass needle spun erratically, seduced by magnetic deposits buried deep within the limestone "
        "canyon. Ahead, the undergrowth gave way to a moss-covered plaza crowned by colossal stone "
        "monoliths carved in the likeness of winged serpents. Vines draped like emerald serpents over "
        "crumbling terraces and sunken stairwells. Maya stepped cautiously onto the first flagstone, "
        "testing its weight. A deep rumble shook the earth, followed by the clanking of ancient "
        "counterweights resetting within the temple walls. Somewhere inside the cavernous doorway, torches "
        "flared to life with blue phosphorus fire, illuminating the path toward the legendary serpent jewel."
    ),
}


def tokenize(text: str, lowercase: bool = True, strip_punct: bool = True) -> list[str]:
    """Tokenize raw text into a list of word tokens.

    Args:
        text: Input string to tokenize.
        lowercase: Whether to convert text to lowercase.
        strip_punct: If True, keeps word tokens only (including words with apostrophes like 'don't').
                     If False, also retains punctuation marks as individual tokens.

    Returns:
        List of cleaned token strings.
    """
    if lowercase:
        text = text.lower()

    if strip_punct:
        # Match words with internal apostrophes/hyphens: e.g. "don't", "seven-echo"
        tokens = re.findall(r"\b[a-z0-9]+(?:[-'][a-z0-9]+)*\b", text if lowercase else text.lower())
    else:
        # Preserve words and basic sentence punctuation
        tokens = re.findall(r"\b[a-z0-9]+(?:[-'][a-z0-9]+)*\b|[.,!?;:]", text)

    return tokens


class Corpus:
    """Represents a text corpus of short stories with tokenization and metadata."""

    def __init__(self, stories: dict[str, str] | None = None, lowercase: bool = True, strip_punct: bool = True):
        self.lowercase = lowercase
        self.strip_punct = strip_punct
        self.stories: dict[str, str] = stories or {}
        self.genre_tokens: dict[str, list[str]] = {}
        self.tokens: list[str] = []
        self._build()

    def _build(self) -> None:
        self.tokens = []
        self.genre_tokens = {}
        for genre, text in self.stories.items():
            genre_toks = tokenize(text, lowercase=self.lowercase, strip_punct=self.strip_punct)
            self.genre_tokens[genre] = genre_toks
            self.tokens.extend(genre_toks)

    @property
    def vocabulary(self) -> set[str]:
        return set(self.tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def genres(self) -> list[str]:
        return sorted(self.stories.keys())

    def get_tokens(self, genre: str | None = None) -> list[str]:
        if genre:
            genre_lower = genre.lower()
            if genre_lower in self.genre_tokens:
                return self.genre_tokens[genre_lower]
            raise ValueError(f"Unknown genre '{genre}'. Available: {', '.join(self.genres)}")
        return list(self.tokens)

    def stats(self) -> dict[str, int | list[str]]:
        return {
            "num_stories": len(self.stories),
            "num_tokens": len(self.tokens),
            "vocab_size": self.vocab_size,
            "genres": self.genres,
        }

    @classmethod
    def load(
        cls,
        corpus_dir: str | Path | None = None,
        genre: str | None = None,
        lowercase: bool = True,
        strip_punct: bool = True,
    ) -> "Corpus":
        """Load corpus from a directory of .txt files, falling back to built-in genre stories."""
        stories: dict[str, str] = {}

        # Attempt to load from directory if provided or if data/corpus exists
        search_path = Path(corpus_dir) if corpus_dir else Path("data/corpus")
        if search_path.is_dir():
            for file_path in search_path.glob("*.txt"):
                story_genre = file_path.stem.lower()
                try:
                    stories[story_genre] = file_path.read_text(encoding="utf-8")
                except Exception:
                    pass

        # Fallback to built-in stories if none loaded
        if not stories:
            stories = dict(BUILTIN_STORIES)

        # Filter by genre if requested
        if genre:
            target = genre.lower()
            if target in stories:
                stories = {target: stories[target]}
            elif target in BUILTIN_STORIES:
                stories = {target: BUILTIN_STORIES[target]}

        return cls(stories=stories, lowercase=lowercase, strip_punct=strip_punct)


# --- Milestone 1: Prepare a text corpus of short stories.
# Hint: Use text preprocessing and tokenization techniques to prepare your corpus.
def milestone_1(
    corpus_dir: str | Path | None = None,
    genre: str | None = None,
) -> Corpus:
    """Milestone 1: Prepare and tokenize a text corpus of short stories.

    Loads the genre corpus, preprocesses and tokenizes the stories, and returns
    a Corpus instance containing the processed tokens and vocabulary.
    """
    corpus = Corpus.load(corpus_dir=corpus_dir, genre=genre)
    return corpus


class NGramModel:
    """N-gram language model supporting frequency counting, MLE, and smoothing."""

    def __init__(self, n: int = 2):
        if n < 1:
            raise ValueError("N-gram order n must be at least 1.")
        self.n: int = n
        self.ngram_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.unigram_counts: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.total_tokens: int = 0

    def _normalize_context(self, context: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
        """Normalize input context to a tuple of length (n - 1)."""
        if context is None:
            return ()
        if isinstance(context, str):
            ctx_tokens = tokenize(context, lowercase=True, strip_punct=True)
        else:
            ctx_tokens = [t.lower() for t in context]

        needed_len = self.n - 1
        if len(ctx_tokens) > needed_len:
            ctx_tokens = ctx_tokens[-needed_len:] if needed_len > 0 else []
        elif len(ctx_tokens) < needed_len:
            # Keep as-is (e.g. for shorter prompts)
            pass

        return tuple(ctx_tokens)

    def train(self, tokens: Sequence[str]) -> "NGramModel":
        """Train the model on a sequence of tokens."""
        token_list = [t.lower() for t in tokens]
        self.vocabulary.update(token_list)
        self.total_tokens += len(token_list)

        for tok in token_list:
            self.unigram_counts[tok] += 1

        context_len = self.n - 1
        if context_len == 0:
            # Unigram model: context is empty tuple ()
            for tok in token_list:
                self.ngram_counts[()][tok] += 1
                self.context_counts[()] += 1
        else:
            for i in range(len(token_list) - context_len):
                ctx = tuple(token_list[i : i + context_len])
                next_word = token_list[i + context_len]
                self.ngram_counts[ctx][next_word] += 1
                self.context_counts[ctx] += 1

        return self

    def count(self, context: tuple[str, ...] | list[str] | str, word: str) -> int:
        """Return the raw count of word occurring after context."""
        ctx = self._normalize_context(context)
        return self.ngram_counts[ctx][word.lower()]

    def context_count(self, context: tuple[str, ...] | list[str] | str) -> int:
        """Return the total occurrences of context."""
        ctx = self._normalize_context(context)
        return self.context_counts[ctx]

    def mle_prob(self, context: tuple[str, ...] | list[str] | str, word: str) -> float:
        """Calculate Maximum Likelihood Estimation (MLE) probability P(word | context)."""
        ctx = self._normalize_context(context)
        denom = self.context_counts[ctx]
        if denom == 0:
            return 0.0
        return self.ngram_counts[ctx][word.lower()] / denom

    def mle_candidates(self, context: tuple[str, ...] | list[str] | str) -> list[tuple[str, float]]:
        """Return list of (word, mle_probability) for all words observed after context."""
        ctx = self._normalize_context(context)
        denom = self.context_counts[ctx]
        if denom == 0:
            return []
        items = [
            (word, count / denom)
            for word, count in self.ngram_counts[ctx].items()
        ]
        items.sort(key=lambda x: (-x[1], x[0]))
        return items

    def smoothed_prob(
        self,
        context: tuple[str, ...] | list[str] | str,
        word: str,
        k: float = 1.0,
    ) -> float:
        """Calculate Add-k (Laplace/Lidstone) smoothed probability P(word | context).

        Formula:
            P(w | context) = (C(context, w) + k) / (C(context) + k * |V|)

        Handles:
            - Unseen n-grams (context seen, word not seen in that context)
            - Unseen contexts (context never seen anywhere in corpus)
            - Words in or out of the training vocabulary
        """
        if k < 0:
            raise ValueError("Smoothing parameter k must be non-negative.")
        if k == 0:
            return self.mle_prob(context, word)

        ctx = self._normalize_context(context)
        v_size = len(self.vocabulary)
        if v_size == 0:
            return 0.0

        word_lower = word.lower()
        word_count = self.ngram_counts[ctx][word_lower]
        ctx_count = self.context_counts[ctx]

        if word_lower in self.vocabulary:
            return (word_count + k) / (ctx_count + k * v_size)
        else:
            return k / (ctx_count + k * (v_size + 1))

    # Alias for convenience
    prob = smoothed_prob

    def get_distribution(
        self,
        context: tuple[str, ...] | list[str] | str,
        k: float = 1.0,
    ) -> dict[str, float]:
        """Compute the full smoothed probability distribution over the vocabulary for context."""
        ctx = self._normalize_context(context)
        v_size = len(self.vocabulary)
        if v_size == 0:
            return {}

        if k == 0:
            denom = self.context_counts[ctx]
            if denom == 0:
                return {w: 1.0 / v_size for w in self.vocabulary}
            return {
                w: self.ngram_counts[ctx][w] / denom
                for w in self.vocabulary
            }

        ctx_count = self.context_counts[ctx]
        denom = ctx_count + k * v_size
        return {
            w: (self.ngram_counts[ctx][w] + k) / denom
            for w in self.vocabulary
        }

    def perplexity(self, tokens: Sequence[str], k: float = 1.0) -> float:
        """Calculate model perplexity on a test token sequence using add-k smoothing."""
        if not tokens or not self.vocabulary:
            return float("inf")

        token_list = [t.lower() for t in tokens]
        context_len = self.n - 1
        log_prob_sum = 0.0
        n_predictions = 0

        if context_len == 0:
            for tok in token_list:
                p = self.smoothed_prob((), tok, k=k)
                log_prob_sum += math.log(max(p, 1e-12))
                n_predictions += 1
        else:
            for i in range(len(token_list) - context_len):
                ctx = tuple(token_list[i : i + context_len])
                next_word = token_list[i + context_len]
                p = self.smoothed_prob(ctx, next_word, k=k)
                log_prob_sum += math.log(max(p, 1e-12))
                n_predictions += 1

        if n_predictions == 0:
            return float("inf")

        return math.exp(-log_prob_sum / n_predictions)


# --- Milestone 2: Implement n-gram counting and maximum-likelihood probabilities.
# Hint: Apply n-gram frequency counting and maximum-likelihood probability calculation.
def milestone_2(
    tokens: Sequence[str] | None = None,
    n: int = 2,
    genre: str | None = None,
) -> NGramModel:
    """Milestone 2: Implement n-gram counting and maximum-likelihood probabilities.

    Trains an NGramModel of order n on the corpus tokens, computing frequency
    counts and maximum-likelihood probabilities.
    """
    if tokens is None:
        corpus = milestone_1(genre=genre)
        tokens = corpus.tokens

    model = NGramModel(n=n)
    model.train(tokens)
    return model


# --- Milestone 3: Add add-k smoothing to handle unseen n-grams.
# Hint: Implement add-k smoothing to manage unseen n-grams and improve model robustness.
def milestone_3(
    model: NGramModel | None = None,
    k: float = 1.0,
    genre: str | None = None,
) -> NGramModel:
    """Milestone 3: Add add-k smoothing to handle unseen n-grams.

    Returns the trained NGramModel equipped with add-k smoothing to handle
    unseen n-grams, unseen contexts, and zero-probability scenarios.
    """
    if model is None:
        model = milestone_2(genre=genre)
    return model


# --- Milestone 4: Develop a function to generate text by sampling from the model.
# Hint: Develop a function for random sampling based on your computed probabilities to generate text.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Create a CLI interface for users to input text and get suggestions.
# Hint: Utilize command-line argument parsing and user input handling to build the CLI.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
