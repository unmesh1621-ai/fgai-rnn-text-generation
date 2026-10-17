"""Temperature sampling and diversity metrics for generated text."""

import torch


@torch.no_grad()
def generate(model, vocab, prompt_tokens, context, n_words, temperature, device,
             generator=None):
    """Autoregressively extend a prompt, feeding back each sampled word."""
    model.eval()
    ids = vocab.encode(prompt_tokens)
    pad = vocab.stoi["<pad>"]
    unk = vocab.stoi["<unk>"]
    new_ids = []
    for _ in range(n_words):
        window = ids[-context:]
        window = [pad] * (context - len(window)) + window
        logits = model(torch.tensor([window], device=device))[0]
        logits[pad] = logits[unk] = float("-inf")
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        ids.append(nxt)
        new_ids.append(nxt)
    return vocab.decode(new_ids)


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(tokens, n):
    grams = ngrams(tokens, n)
    return len(set(grams)) / len(grams) if grams else 0.0


def repeated_trigram_rate(tokens):
    """Share of trigrams that already appeared earlier in the same text."""
    grams = ngrams(tokens, 3)
    seen, repeats = set(), 0
    for g in grams:
        repeats += g in seen
        seen.add(g)
    return repeats / len(grams) if grams else 0.0


def vocab_coverage(tokens, train_counts):
    """Fraction of generated words that are rare (< 10 uses) in training."""
    if not tokens:
        return 0.0
    return sum(train_counts.get(t, 0) < 10 for t in tokens) / len(tokens)
