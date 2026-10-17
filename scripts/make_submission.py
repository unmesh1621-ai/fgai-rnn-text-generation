"""Build the submission Word document from the files in results/.
Run from the repo root:  python scripts/make_submission.py
"""

import csv
import json

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

OUT = "Unmesh_230911030_FGAI_Assignment2.docx"
REPO = "https://github.com/unmesh1621-ai/fgai-rnn-text-generation"


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
    for row in rows:
        cells = t.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = str(text)
    for row in t.rows:
        for i, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
            if widths:
                cell.width = Cm(widths[i])
    return t


def main():
    stats = json.load(open("results/dataset_stats.json"))
    test = list(csv.DictReader(open("results/test_metrics.csv")))
    temps = list(csv.DictReader(open("results/temperature_metrics.csv")))

    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(1.8)
        s.left_margin = s.right_margin = Cm(2)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    title = doc.add_heading("Foundations of Generative AI (ICT4403) - Assignment 2", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("Next-Word Prediction and Temperature-Controlled Text Generation: "
                            "Simple RNN vs Bidirectional RNN vs LSTM vs GRU")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].italic = True

    table(doc, ["Field", "Details"], [
        ["Student Name", "Unmesh"],
        ["Registration Number", "230911030"],
        ["Course", "Foundations of Generative AI (ICT4403)"],
        ["Dataset", "A Study in Scarlet - Arthur Conan Doyle (Project Gutenberg #244)"],
        ["Dataset Link", "https://www.gutenberg.org/ebooks/244"],
        ["GitHub Repository", REPO],
    ], widths=[4.5, 12.5])

    doc.add_heading("1. Objective", level=2)
    doc.add_paragraph(
        "Train four word-level recurrent language models that predict the next word from "
        "the previous 10 words, compare them on held-out text, and study how sampling "
        "temperature changes the quality and diversity of generated text.")

    doc.add_heading("2. Data and Method", level=2)
    for line in [
        f"Corpus: {stats['total_tokens']:,} tokens after removing the Gutenberg header/footer, "
        "lowercasing and splitting words and punctuation.",
        f"Contiguous 80/10/10 split ({stats['train_tokens']:,} / {stats['val_tokens']:,} / "
        f"{stats['test_tokens']:,} tokens), so no window appears in two splits.",
        f"Vocabulary built from the training split only (words seen at least 3 times): "
        f"{stats['vocab_size']:,} entries; {stats['test_unk_rate']:.1%} of test tokens are unknown.",
        "Shared architecture: Embedding(256) -> recurrent core (256 units) -> LayerNorm -> "
        "Dropout(0.4) -> Linear(vocab). The BiRNN averages both directions over the window.",
        "Training: AdamW (lr 2e-3), batch 128, gradient clipping 0.5, learning-rate halving "
        "on plateau, early stopping (patience 3); the best validation checkpoint is tested.",
        "Hyperparameters were tuned on validation loss only; the test set was used once.",
    ]:
        doc.add_paragraph(line, style="List Bullet")

    doc.add_heading("3. Test Results", level=2)
    table(doc, ["Model", "Params", "Best epoch", "Loss", "Perplexity", "Top-1", "Top-5"], [
        [r["model"], f"{int(r['params']):,}", r["best_epoch"], f"{float(r['loss']):.3f}",
         f"{float(r['perplexity']):.1f}", f"{float(r['top1_acc']):.1%}",
         f"{float(r['top5_acc']):.1%}"] for r in test])
    doc.add_picture("results/learning_curves.png", width=Cm(16))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("4. Temperature Analysis", level=2)
    doc.add_paragraph(
        "Each model continued three prompts for 40 words at T = 0.3, 0.7, 1.0 and 1.3. "
        "Only newly generated words are scored. Distinct-n is unique n-grams divided by total "
        "n-grams; repeated trigrams is the share of trigrams already seen in the same output.")
    table(doc, ["Model", "T", "Distinct-1", "Distinct-2", "Repeated trigrams"], [
        [r["model"], r["temperature"], f"{float(r['distinct_1']):.3f}",
         f"{float(r['distinct_2']):.3f}", f"{float(r['repeated_trigrams']):.3f}"]
        for r in temps])
    doc.add_picture("results/temperature_effect.png", width=Cm(16))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("5. Conclusions", level=2)
    best = min(test, key=lambda r: float(r["perplexity"]))
    for line in [
        f"{best['model']} is the best model, with test perplexity {float(best['perplexity']):.1f} "
        f"and top-1 accuracy {float(best['top1_acc']):.1%}. Gating helps keep useful context "
        "across the window.",
        "The Simple RNN came close to the gated models: on a short 10-word window the "
        "vanishing-gradient problem is mild.",
        "The Bidirectional RNN has the most parameters but the worst scores. Reading the window "
        "backwards adds little when the last few words matter most, and it overfits sooner. It "
        "stays autoregressive because its backward pass only reads past words in the window.",
        "Low temperature (0.3) gives locally fluent but looping text; 0.7 gives the best balance; "
        "1.0-1.3 maximises diversity but breaks grammar. Diversity metrics must be read together "
        "with perplexity and the actual samples.",
    ]:
        doc.add_paragraph(line, style="List Bullet")

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
