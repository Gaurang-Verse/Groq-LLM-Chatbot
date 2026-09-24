"""
benchmark.py

Measures response latency for each model supported by the Groq LLM Chatbot.
Run this from the project root after installing requirements.txt.

Usage:
    python benchmark.py
"""

import os
import time
import json
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

TEST_PROMPTS = [
    "Explain what a hash map is in two sentences.",
    "Write a haiku about the ocean.",
    "What is the capital of France?",
]

NUM_RUNS_PER_PROMPT = 3  # run each prompt 3x per model, average the result


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Check your .env file.")
    return Groq(api_key=api_key)


def benchmark_model(client: Groq, model: str) -> dict:
    """Run all test prompts against a model and return timing stats."""
    first_token_times = []
    total_times = []
    output_token_counts = []

    for prompt in TEST_PROMPTS:
        for _ in range(NUM_RUNS_PER_PROMPT):
            start = time.perf_counter()
            first_token_time = None
            token_count = 0

            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
                stream=True,
            )

            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start
                    token_count += 1

            total_time = time.perf_counter() - start
            first_token_times.append(first_token_time)
            total_times.append(total_time)
            output_token_counts.append(token_count)

    return {
        "model": model,
        "avg_time_to_first_token_sec": round(mean(first_token_times), 3),
        "avg_total_response_time_sec": round(mean(total_times), 3),
        "avg_output_tokens": round(mean(output_token_counts), 1),
        "avg_tokens_per_sec": round(
            mean(output_token_counts) / mean(total_times), 1
        ),
        "runs": len(total_times),
    }


def main():
    client = get_client()
    results = []

    for model in MODELS:
        print(f"Benchmarking {model} ...")
        try:
            result = benchmark_model(client, model)
            results.append(result)
            print(f"  Done: {result}")
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({"model": model, "error": str(e)})

    output_path = PROJECT_ROOT / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}")
    print("\n--- Summary ---")
    for r in results:
        if "error" in r:
            print(f"{r['model']}: FAILED ({r['error']})")
        else:
            print(
                f"{r['model']}: "
                f"{r['avg_tokens_per_sec']} tok/s, "
                f"{r['avg_time_to_first_token_sec']}s to first token, "
                f"{r['avg_total_response_time_sec']}s total avg"
            )


if __name__ == "__main__":
    main()