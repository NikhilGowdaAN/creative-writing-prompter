"""Command-line interface for the Creative Writing Prompter.

Usage:
    python -m app.cli --help
    python -m app.cli hello
    python -m app.cli suggest "The ancient" [--genre fantasy] [--top 5] [--temperature 1.0]
    python -m app.cli generate "The dark citadel" [--length 25] [--genre horror]
    python -m app.cli corpus [--genre scifi]
    python -m app.cli interactive [--genre fantasy]
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from app.core import Corpus, NGramModel, milestone_1, milestone_2, milestone_4, tokenize
from app.storage import Store


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="creative-writing-prompter",
        description=(
            "Aspiring writers use this CLI tool to get context-aware next-word suggestions, "
            "helping them develop stories and overcome writer's block by sampling from a genre corpus."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Hello command (plumbing check)
    sub.add_parser("hello", help="check that everything is wired up")

    # 2. Suggest command
    p_suggest = sub.add_parser("suggest", help="get context-aware next-word suggestions for a writing prompt")
    p_suggest.add_argument("prompt", type=str, help="current text or sentence fragment written so far")
    p_suggest.add_argument("-g", "--genre", type=str, default=None, help="genre corpus to sample from (fantasy, scifi, mystery, horror, adventure)")
    p_suggest.add_argument("-n", "--ngram", type=int, default=2, help="n-gram model order (e.g., 2 for bigram, 3 for trigram)")
    p_suggest.add_argument("-k", "--smoothing", type=float, default=1.0, help="add-k smoothing parameter (default: 1.0)")
    p_suggest.add_argument("-t", "--temperature", type=float, default=1.0, help="sampling temperature (lower is more deterministic, higher is more diverse)")
    p_suggest.add_argument("--top", type=int, default=5, help="number of next-word suggestions to display (default: 5)")
    p_suggest.add_argument("--save", action="store_true", help="save prompt and suggestions to history store")

    # 3. Generate command
    p_gen = sub.add_parser("generate", help="generate continuous story text to break through writer's block")
    p_gen.add_argument("prompt", type=str, help="starting sentence or prompt for the story")
    p_gen.add_argument("-l", "--length", type=int, default=20, help="number of words to generate (default: 20)")
    p_gen.add_argument("-g", "--genre", type=str, default=None, help="genre corpus (fantasy, scifi, mystery, horror, adventure)")
    p_gen.add_argument("-n", "--ngram", type=int, default=2, help="n-gram order (default: 2)")
    p_gen.add_argument("-k", "--smoothing", type=float, default=1.0, help="add-k smoothing parameter (default: 1.0)")
    p_gen.add_argument("-t", "--temperature", type=float, default=1.0, help="sampling temperature (default: 1.0)")
    p_gen.add_argument("--top-k", type=int, default=20, help="top-k sampling pool limit (default: 20)")
    p_gen.add_argument("--seed", type=int, default=None, help="random seed for reproducible generation")
    p_gen.add_argument("--save", action="store_true", help="save generated story to store")

    # 4. Corpus command
    p_corpus = sub.add_parser("corpus", help="inspect the loaded genre corpus and vocabulary stats")
    p_corpus.add_argument("-g", "--genre", type=str, default=None, help="filter by specific genre")

    # 5. Interactive writing assistant
    p_interactive = sub.add_parser("interactive", help="start an interactive writing session with real-time suggestions")
    p_interactive.add_argument("-g", "--genre", type=str, default=None, help="genre corpus to guide your story")
    p_interactive.add_argument("-n", "--ngram", type=int, default=2, help="n-gram order (default: 2)")
    p_interactive.add_argument("-k", "--smoothing", type=float, default=1.0, help="smoothing parameter (default: 1.0)")
    p_interactive.add_argument("-t", "--temperature", type=float, default=1.0, help="temperature (default: 1.0)")

    return parser


def handle_suggest(args: argparse.Namespace) -> None:
    corpus = milestone_1(genre=args.genre)
    model = NGramModel(n=args.ngram)
    model.train(corpus.tokens)

    suggestions = model.suggest_next_words(
        context=args.prompt,
        num_suggestions=args.top,
        k=args.smoothing,
        temperature=args.temperature,
    )

    print("\n" + "=" * 60)
    print("  CREATIVE WRITING PROMPTER - NEXT-WORD SUGGESTIONS")
    print("=" * 60)
    genre_display = args.genre.capitalize() if args.genre else "All Genres (Combined)"
    print(f"Genre:        {genre_display}")
    print(f"N-Gram Model: {args.ngram}-gram (smoothing k={args.smoothing}, temp={args.temperature})")
    print(f"Prompt:       \"{args.prompt}\"")
    print("-" * 60)
    print(f"{'#':<4} {'SUGGESTION':<20} {'PROBABILITY':<12} {'VISUAL'}")
    print("-" * 60)

    for i, (word, prob) in enumerate(suggestions, start=1):
        pct = prob * 100
        bar = "█" * int(min(pct, 100) / 4)
        print(f"[{i}] {word:<20} {pct:6.2f}%      {bar}")

    print("-" * 60)

    # Generated sample completion
    sample_completion = model.generate(
        prompt=args.prompt,
        max_words=10,
        k=args.smoothing,
        temperature=args.temperature,
    )
    print(f"Sample Continuation: \"{sample_completion} ...\"\n")

    if args.save:
        store = Store()
        history = store.get("suggestions_history", [])
        history.append({
            "prompt": args.prompt,
            "genre": args.genre,
            "suggestions": [(w, float(p)) for w, p in suggestions],
            "sample_completion": sample_completion,
        })
        store.put("suggestions_history", history)
        print("✓ Saved suggestions to session history.\n")


def handle_generate(args: argparse.Namespace) -> None:
    corpus = milestone_1(genre=args.genre)
    model = NGramModel(n=args.ngram)
    model.train(corpus.tokens)

    text = model.generate(
        prompt=args.prompt,
        max_words=args.length,
        k=args.smoothing,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("  STORY INSPIRATION GENERATOR")
    print("=" * 60)
    genre_display = args.genre.capitalize() if args.genre else "All Genres"
    print(f"Genre:  {genre_display} | Model: {args.ngram}-gram (T={args.temperature})")
    print("-" * 60)
    print(f"\n{text}\n")
    print("=" * 60 + "\n")

    if args.save:
        store = Store()
        drafts = store.get("drafts", [])
        drafts.append({
            "prompt": args.prompt,
            "genre": args.genre,
            "generated_text": text,
        })
        store.put("drafts", drafts)
        print("✓ Saved generated passage to store.\n")


def handle_corpus(args: argparse.Namespace) -> None:
    corpus = milestone_1(genre=args.genre)
    stats = corpus.stats()

    print("\n" + "=" * 60)
    print("  GENRE CORPUS DETAILS")
    print("=" * 60)
    print(f"Stories Loaded:   {stats['num_stories']}")
    print(f"Available Genres: {', '.join(stats['genres'])}")
    print(f"Total Tokens:     {stats['num_tokens']}")
    print(f"Vocabulary Size:  {stats['vocab_size']}")
    print("=" * 60 + "\n")


def handle_interactive(args: argparse.Namespace) -> None:
    corpus = milestone_1(genre=args.genre)
    model = NGramModel(n=args.ngram)
    model.train(corpus.tokens)

    genre_name = args.genre.capitalize() if args.genre else "Multi-Genre"
    print("\n" + "=" * 65)
    print(f"  CREATIVE WRITING ASSISTANT (Interactive Mode - {genre_name})")
    print("=" * 65)
    print("Commands:")
    print("  Type your story words or sentences.")
    print("  Type '1'-'5' to select one of the top suggestions.")
    print("  Type '/gen' to auto-generate the next 15 words.")
    print("  Type '/exit' or '/quit' to finish your session.")
    print("-" * 65 + "\n")

    current_story: list[str] = []

    try:
        while True:
            prompt_context = " ".join(current_story) if current_story else "the"
            suggestions = model.suggest_next_words(
                context=prompt_context,
                num_suggestions=5,
                k=args.smoothing,
                temperature=args.temperature,
            )

            print("Current story: " + (" ".join(current_story) if current_story else "[Start typing below]"))
            sugg_line = " | ".join(f"[{i+1}] {w} ({p*100:.1f}%)" for i, (w, p) in enumerate(suggestions))
            print(f"Suggestions: {sugg_line}")

            user_input = input("Write > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit"):
                print("\nEnding session. Your story so far:\n")
                print(" ".join(current_story) if current_story else "(No text written)")
                print("\nKeep writing!\n")
                break

            if user_input.lower() == "/gen":
                gen_text = model.generate(
                    prompt=prompt_context,
                    max_words=15,
                    k=args.smoothing,
                    temperature=args.temperature,
                )
                current_story = gen_text.split()
                print(f"\nGenerated: {gen_text}\n")
                continue

            if user_input in ("1", "2", "3", "4", "5"):
                idx = int(user_input) - 1
                if idx < len(suggestions):
                    picked = suggestions[idx][0]
                    current_story.append(picked)
                    continue

            # Writer entered new words
            new_tokens = tokenize(user_input, lowercase=False, strip_punct=True)
            current_story.extend(new_tokens)
            print()

    except (KeyboardInterrupt, EOFError):
        print("\nSession ended.")


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Creative Writing Prompter is ready. Open README.md for your milestones.")
    elif args.command == "suggest":
        handle_suggest(args)
    elif args.command == "generate":
        handle_generate(args)
    elif args.command == "corpus":
        handle_corpus(args)
    elif args.command == "interactive":
        handle_interactive(args)


if __name__ == "__main__":
    main()
