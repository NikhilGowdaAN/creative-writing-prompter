# Creative Writing Prompter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/8zc24u985n/badge.svg)](https://ask.thili.ai/b/8zc24u985n)

This Creative Writing Prompter is a CLI tool designed for aspiring writers. It leverages a next-word prediction engine, built from scratch, to analyze a corpus of short stories and provide contextually relevant word suggestions. Overcome writer's block and develop your stories with intelligent, genre-aware prompts.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that provides context-aware next-word suggestions for aspiring writers, helping them overcome writer's block. The core is a next-word prediction engine that samples from a genre corpus.

**Done when.** The tool successfully prepares a text corpus, implements n-gram counting with add-k smoothing, generates text by sampling from the model, and offers a functional CLI for users to input text and receive suggestions.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare a text corpus of short stories.**  
      Hint: Use text preprocessing and tokenization techniques to prepare your corpus.
- [x] **2. Implement n-gram counting and maximum-likelihood probabilities.**  
      Hint: Apply n-gram frequency counting and maximum-likelihood probability calculation.
- [ ] **3. Add add-k smoothing to handle unseen n-grams.**  
      Hint: Implement add-k smoothing to manage unseen n-grams and improve model robustness.
- [ ] **4. Develop a function to generate text by sampling from the model.**  
      Hint: Develop a function for random sampling based on your computed probabilities to generate text.
- [ ] **5. Create a CLI interface for users to input text and get suggestions.**  
      Hint: Utilize command-line argument parsing and user input handling to build the CLI.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify the n-gram size (e.g., bigram, trigram) at runtime for different prediction behaviors.
- Implement a 'temperature' parameter for text generation to control the randomness of suggestions, making them more or less predictable.
- Display multiple next-word suggestions with their probabilities instead of just one, giving users more options.
