"""A semantic cache that hits across languages (keys are code prefixes)."""

from sphota.usecases import SemanticCache

cache = SemanticCache(depth=2)


def expensive_llm_call(question: str) -> str:
    return f"(answer generated for: {question})"


for q in [
    "How do I reset my password?",
    "¿Cómo restablezco mi contraseña?",  # Spanish, same meaning
    "How can I change my password?",  # paraphrase
    "What is the capital of France?",
]:
    before = cache.stats.hits
    answer = cache.get_or_compute(q, expensive_llm_call)
    print(f"{'HIT ' if cache.stats.hits > before else 'MISS'} {q!r} → {answer}")
print(f"hit rate {cache.stats.hit_rate:.0%}")
