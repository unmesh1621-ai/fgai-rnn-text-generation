# Next-Word Prediction with Recurrent Networks: RNN vs BiRNN vs LSTM vs GRU

Foundations of Generative AI (ICT4403) - Assignment 2

- **Student:** Unmesh
- **Registration Number:** 230911030
- **Dataset:** *A Study in Scarlet* by Arthur Conan Doyle - [Project Gutenberg #244](https://www.gutenberg.org/ebooks/244)
- **Repository:** https://github.com/unmesh1621-ai/fgai-rnn-text-generation

## Overview

This project trains four word-level recurrent language models to predict the
next word from the previous 10 words, compares them on held-out text, and uses
each model to generate new text at different sampling temperatures.

The models are:

| Model | Recurrent core | Summary vector fed to the classifier |
| --- | --- | --- |
| Simple RNN | `nn.RNN` (tanh) | hidden output at the last time step |
| Bidirectional RNN | `nn.RNN(bidirectional=True)` | mean over time of forward + backward outputs |
| LSTM | `nn.LSTM` | hidden output at the last time step |
| GRU | `nn.GRU` | hidden output at the last time step |

All four share the same surrounding layers, so the recurrent core is the only thing
that changes:

```
10 word ids -> Embedding(128) -> Dropout -> Recurrent core (256 units)
            -> summary vector -> LayerNorm -> Dropout -> Linear(vocab) -> softmax
```

## Dataset and preprocessing

1. The Project Gutenberg licence header and footer are removed.
2. The text is lowercased and split into words and punctuation marks
   (`. , ! ? ; :`) with a regular expression. Words keep internal apostrophes (`don't`).
3. The token stream is split **in order** into 80% train, 10% validation and 10% test.
   A contiguous split means no 10-word window can appear in two splits.
4. The vocabulary is built from the **training split only**, keeping words seen at
   least 3 times. Other words map to `<unk>`.
5. Every 10-word window is paired with the word that follows it (the target).

| Statistic | Value |
| --- | --- |
| Total tokens | 49,681 |
| Train / val / test tokens | 39,744 / 4,968 / 4,969 |
| Vocabulary size (incl. `<pad>`, `<unk>`) | 1,543 |
| Test tokens that are `<unk>` | 12.2% |
| Train / val / test windows | 39,734 / 4,958 / 4,959 |

The 12% unknown rate in the test split is high because the last part of the book
introduces new names and places. This makes the task harder but keeps the
evaluation honest, since nothing from the test text leaks into the vocabulary.

## Training setup

All models use the same settings and the same random seed (1621):

| Setting | Value |
| --- | --- |
| Context length | 10 words |
| Embedding / hidden size | 128 / 256 |
| Dropout | 0.3 |
| Optimiser | AdamW, lr 2e-3, weight decay 1e-4 |
| LR schedule | ReduceLROnPlateau (halve after 1 epoch without improvement) |
| Batch size | 128 |
| Gradient clipping | max norm 0.5 |
| Max epochs / early stopping | 20 / stop after 3 epochs without validation improvement |

The weights from the epoch with the lowest validation loss are kept and evaluated
on the test set.

## Results

### Test set

| Model | Parameters | Best epoch | Test loss | Perplexity | Top-1 acc | Top-5 acc | Train time (s, CPU) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple RNN | 693,383 | 6 | 4.213 | 67.5 | 19.8% | 46.0% | 28.8 |
| Bidirectional RNN | 1,187,719 | 6 | 4.286 | 72.7 | 19.0% | 44.1% | 46.8 |
| **LSTM** | 989,831 | 3 | **4.144** | **63.1** | 20.4% | 46.6% | 33.4 |
| GRU | 891,015 | 4 | 4.181 | 65.4 | **21.1%** | **47.0%** | 40.3 |

![Learning curves](results/learning_curves.png)
![Test metrics](results/test_metrics.png)

**Observations**

- The gated models (LSTM, GRU) beat the plain RNN on every metric. Their gates help
  them keep useful information across the 10-word window.
- LSTM has the lowest perplexity. GRU is close behind and slightly more accurate,
  with about 10% fewer parameters.
- The Bidirectional RNN has the most parameters but the worst scores. Reading the
  window backwards adds little for next-word prediction, because the most useful
  signal is the last few words. The extra capacity mostly leads to overfitting.
- All models overfit quickly on a 40k-word corpus. Training loss keeps falling
  while validation loss bottoms out after 3-6 epochs, which is why early stopping
  matters here.

### Temperature analysis

Each model continues three prompts (`"it was the"`, `"sherlock holmes said that"`,
`"the man who had"`) for 40 new words at temperatures 0.3, 0.7, 1.0 and 1.3.
Only the newly generated words are scored:

- **Distinct-1 / Distinct-2:** unique unigrams / bigrams divided by the total (higher means more varied).
- **Repeated trigrams:** share of trigrams that already appeared earlier in the same output (higher means more looping).
- **Rare-word share:** share of generated words that appear fewer than 10 times in training.

| Model | T | Distinct-1 | Distinct-2 | Repeated trigrams | Rare-word share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple RNN | 0.3 | 0.575 | 0.838 | 0.070 | 0.050 |
| Simple RNN | 0.7 | 0.742 | 0.949 | 0.009 | 0.092 |
| Simple RNN | 1.0 | 0.758 | 0.974 | 0.000 | 0.158 |
| Simple RNN | 1.3 | 0.858 | 0.992 | 0.000 | 0.242 |
| Bidirectional RNN | 0.3 | 0.500 | 0.727 | 0.132 | 0.217 |
| Bidirectional RNN | 0.7 | 0.758 | 0.966 | 0.000 | 0.142 |
| Bidirectional RNN | 1.0 | 0.800 | 1.000 | 0.000 | 0.192 |
| Bidirectional RNN | 1.3 | 0.858 | 1.000 | 0.000 | 0.317 |
| LSTM | 0.3 | 0.475 | 0.675 | 0.228 | 0.033 |
| LSTM | 0.7 | 0.692 | 0.923 | 0.009 | 0.042 |
| LSTM | 1.0 | 0.792 | 0.974 | 0.009 | 0.142 |
| LSTM | 1.3 | 0.842 | 1.000 | 0.000 | 0.258 |
| GRU | 0.3 | 0.600 | 0.855 | 0.044 | 0.075 |
| GRU | 0.7 | 0.700 | 0.940 | 0.000 | 0.117 |
| GRU | 1.0 | 0.758 | 0.957 | 0.009 | 0.142 |
| GRU | 1.3 | 0.858 | 0.992 | 0.000 | 0.225 |

![Temperature effect](results/temperature_effect.png)

Sample outputs (LSTM):

> **T = 0.3** - *the man who had* been a large of the same, and the door, and the other was a man of the man who had been a large of the case.

> **T = 0.7** - *the man who had* been it? i asked. no one, and i asked. it is up on the case and you? she asked. sherlock holmes, and there was a man who...

> **T = 1.3** - *the man who had* been warm and resolute before they made our you began and look to above from the mouth entered the valley of my him has.

**Observations**

- **Low temperature (0.3)** sharpens the distribution towards the most likely words.
  Output is grammatical in short stretches but falls into loops ("a man of the room,
  and he was a man of the door"). The LSTM, as the most confident model, repeats
  itself the most (23% repeated trigrams).
- **Medium temperature (0.7)** gives the best balance. Loops almost disappear, and
  the text still reads like the book (dialogue, "i asked", "sherlock holmes").
- **High temperature (1.0-1.3)** flattens the distribution. Diversity hits its
  maximum (Distinct-2 near 1.0) and rare words like "afghanistan", "utah" and
  "jefferson hope" show up, but grammar breaks down.
- Diversity metrics alone reward the incoherent high-temperature text. They have
  to be read alongside perplexity and the actual samples.

All generated samples are in [`results/samples.txt`](results/samples.txt).

## Why the Bidirectional RNN can still generate text

A bidirectional network normally sees words on both sides of a position, which
would break left-to-right generation. Here the BiRNN only ever reads the
10-word context window. Its backward pass runs over those same 10 past words, never
over future ones, and the model predicts the word *after* the window. Generation
therefore stays autoregressive: each sampled word is appended and the window slides
forward by one.

## Project structure

```
fgai-rnn-text-generation/
├── main.py                 # full pipeline: data -> train 4 models -> evaluate -> sample -> plots
├── textgen/
│   ├── data.py             # Gutenberg cleanup, tokeniser, vocabulary, windowed datasets
│   ├── models.py           # NextWordModel (rnn / birnn / lstm / gru)
│   ├── training.py         # training loop with early stopping, evaluation metrics
│   └── sampling.py         # temperature sampling, diversity metrics
├── data/a_study_in_scarlet.txt
├── results/                # metrics CSVs, plots, samples, best checkpoints (*.pt)
├── train_log.txt           # console log of the run that produced results/
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python main.py            # full run, about 3 minutes on a laptop CPU
python main.py --quick    # 2-epoch smoke test
```

Useful options: `--context`, `--epochs`, `--batch`, `--gen-words`, `--seed`, `--out`.
