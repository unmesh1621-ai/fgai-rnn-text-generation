"""Train and compare RNN, BiRNN, LSTM and GRU next-word models, then sample
text at several temperatures. Run:  python main.py  (add --quick for a smoke test)
"""

import argparse
import csv
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from textgen.data import (Vocab, detokenize, load_book, make_windows,
                          split_chronologically, tokenize)
from textgen.models import NextWordModel, count_params
from textgen.sampling import (distinct_n, generate, repeated_trigram_rate,
                              vocab_coverage)
from textgen.training import evaluate, train_model

MODELS = {"rnn": "Simple RNN", "birnn": "Bidirectional RNN",
          "lstm": "LSTM", "gru": "GRU"}
TEMPERATURES = [0.3, 0.7, 1.0, 1.3]
PROMPTS = ["it was the", "sherlock holmes said that", "the man who had"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_curves(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, h in histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], "--", label=f"{MODELS[kind]} train")
        axes[0].plot(ep, h["val_loss"], label=f"{MODELS[kind]} val")
        axes[1].plot(ep, h["val_top1"], label=MODELS[kind])
    axes[0].set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation top-1 accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bars(rows, path):
    names = [r["model"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(axes, ["perplexity", "top1_acc", "top5_acc"],
                              ["Test perplexity (lower is better)",
                               "Test top-1 accuracy", "Test top-5 accuracy"]):
        ax.bar(names, [r[key] for r in rows], color="#4C72B0")
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_temperature(temp_rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label in dict.fromkeys(r["model"] for r in temp_rows):
        rows = [r for r in temp_rows if r["model"] == label]
        t = [r["temperature"] for r in rows]
        axes[0].plot(t, [r["distinct_2"] for r in rows], "o-", label=label)
        axes[1].plot(t, [r["repeated_trigrams"] for r in rows], "o-", label=label)
    axes[0].set(title="Distinct-2 vs temperature", xlabel="temperature")
    axes[1].set(title="Repeated trigram rate vs temperature", xlabel="temperature")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/a_study_in_scarlet.txt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--context", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--gen-words", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1621)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--tie", action="store_true", help="tie input/output embeddings")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated subset")
    ap.add_argument("--no-test", action="store_true",
                    help="tuning mode: report validation only, skip test/sampling")
    ap.add_argument("--quick", action="store_true", help="2 epochs, tiny run")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.gen_words = 2, 15

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens = tokenize(load_book(args.data))
    train_tok, val_tok, test_tok = split_chronologically(tokens)
    vocab = Vocab(train_tok, min_count=3)
    train_ds, val_ds, test_ds = (make_windows(vocab.encode(t), args.context)
                                 for t in (train_tok, val_tok, test_tok))
    unk_rate = sum(i == 1 for i in vocab.encode(test_tok)) / len(test_tok)
    stats = {
        "total_tokens": len(tokens), "train_tokens": len(train_tok),
        "val_tokens": len(val_tok), "test_tokens": len(test_tok),
        "vocab_size": len(vocab), "test_unk_rate": round(unk_rate, 4),
        "train_windows": len(train_ds), "val_windows": len(val_ds),
        "test_windows": len(test_ds), "context": args.context,
        "most_common_train": vocab.counts.most_common(10),
    }
    print(json.dumps(stats, indent=1))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512)
    test_dl = DataLoader(test_ds, batch_size=512)

    histories, rows, temp_rows, samples = {}, [], [], []
    for kind in args.models.split(","):
        label = MODELS[kind]
        set_seed(args.seed)
        model = NextWordModel(kind, len(vocab), emb_dim=args.emb, hidden=args.hidden,
                              dropout=args.dropout, layers=args.layers,
                              tie_weights=args.tie)
        print(f"\n== {label}  ({count_params(model):,} parameters)")
        hist = train_model(model, train_dl, val_dl, device, epochs=args.epochs,
                           lr=args.lr, weight_decay=args.weight_decay)
        histories[kind] = hist
        if args.no_test:
            print(f"VAL {label} {min(hist['val_loss']):.4f}")
            continue
        torch.save(model.state_dict(), os.path.join(args.out, f"{kind}.pt"))

        test = evaluate(model, test_dl, device)
        rows.append({"model": label, "params": count_params(model),
                     "best_epoch": hist["best_epoch"],
                     "train_seconds": round(hist["train_seconds"], 1),
                     **{k: round(v, 4) for k, v in test.items()}})
        print(f"  test: {rows[-1]}")

        gen = torch.Generator().manual_seed(args.seed)
        for temp in TEMPERATURES:
            all_new = []
            for prompt in PROMPTS:
                new = generate(model, vocab, prompt.split(), args.context,
                               args.gen_words, temp, device, gen)
                all_new.append(new)
                samples.append(f"[{label} | T={temp}] {prompt} >> {detokenize(new)}")
            flat = [w for s in all_new for w in s]
            temp_rows.append({
                "model": label, "temperature": temp,
                "distinct_1": round(np.mean([distinct_n(s, 1) for s in all_new]), 4),
                "distinct_2": round(np.mean([distinct_n(s, 2) for s in all_new]), 4),
                "repeated_trigrams": round(np.mean([repeated_trigram_rate(s) for s in all_new]), 4),
                "rare_word_share": round(vocab_coverage(flat, vocab.counts), 4),
            })

    if args.no_test:
        return
    with open(os.path.join(args.out, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out, "histories.json"), "w") as f:
        json.dump(histories, f, indent=2)
    for name, data in [("test_metrics.csv", rows), ("temperature_metrics.csv", temp_rows)]:
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    with open(os.path.join(args.out, "samples.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(samples) + "\n")

    plot_curves(histories, os.path.join(args.out, "learning_curves.png"))
    plot_bars(rows, os.path.join(args.out, "test_metrics.png"))
    plot_temperature(temp_rows, os.path.join(args.out, "temperature_effect.png"))
    print(f"\nDone. Results in {args.out}/")


if __name__ == "__main__":
    main()
