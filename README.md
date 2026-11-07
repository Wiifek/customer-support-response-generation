# Customer-Support Response Generation with a GRU Encoder–Decoder

NLP course project: text generation with an encoder–decoder neural network **implemented and trained from scratch** in PyTorch (no pretrained generative model).

**Author:** Wifek Chabir

## 1. Task

Given a customer's message, generate the support agent's reply.

| Input | Output |
|---|---|
| `i want assistance to swap some items of order <order_number>` | a complete, polite support reply asking for the details needed to proceed |

## 2. What the project does

Two models are trained and compared. Everything is identical between them except attention.

| Model | Description |
|---|---|
| **Baseline** | Embedding → 1-layer GRU encoder → GRU decoder → Linear → softmax. The encoder's final hidden state initializes the decoder. |
| **Baseline + attention** | Same model, plus additive (Bahdanau-style) attention: at each step the decoder weights all encoder states and receives a context vector. |

Pipeline: text preprocessing → subword vocabulary → tokenization → embeddings → encoder → decoder → training → generation → evaluation.

## 3. Dataset

- **Source:** [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (Hugging Face), loaded with the `datasets` library.
- **Size:** 26,872 (input, response) pairs.
- **Fields used:** `instruction` (customer message, input), `response` (agent reply, target); `intent` is used only to stratify the split.
- **Split (80/10/10, stratified by intent, seed 42):** 21,497 train / 2,687 validation / 2,688 test.

## 4. Preprocessing

- Lowercasing and whitespace normalization.
- Placeholders such as `{{Order Number}}` are converted to single tokens (`<order_number>`; 314 distinct placeholders).
- Shared **SentencePiece BPE** tokenizer, vocabulary size 4,000, trained on the training split only.
- Special tokens: `<PAD>`=0, `<UNK>`=1, `<SOS>`=2, `<EOS>`=3.
- Truncation: inputs to 24 tokens, targets to 256 tokens (95.6% of responses are kept complete).
- Padding with `<PAD>`: ignored by the loss, skipped by the encoder (packed sequences), masked in attention.

## 5. Architecture and hyperparameters

| Setting | Value |
|---|---|
| Embedding dimension | 128 |
| Hidden dimension | 256 |
| Layers | 1 (unidirectional GRU) |
| Dropout | 0.2 |
| Parameters | 2,644,896 (baseline) / 3,997,088 (attention) |
| Loss | Cross-entropy, ignoring `<PAD>` |
| Optimizer | Adam, learning rate 1e-3, gradient clipping 1.0 |
| Batch size / epochs | 64 / 20 |
| Teacher forcing | ratio 0.8 during training (1.0 for validation loss) |
| Decoding | Greedy; stops at `<EOS>` or 254 tokens |

## 6. Results (test set)

| Model | Test PPL | Token acc. | BLEU | ROUGE-1 | ROUGE-L | Train time |
|---|---|---|---|---|---|---|
| GRU (baseline) | 2.995 | 71.3% | 30.8 | 55.2 | 41.7 | 42 min |
| GRU + attention | 2.919 | 72.1% | 34.1 | 57.4 | 43.9 | 92 min |

Attention improves every metric, moderately, at about 2.2× the training time. See the notebook and the report for the learning curves, generated examples, failure analysis and limitations.

## 7. Repository content

```
.
├── NLP_Project_Customer_support_response_generation.ipynb  
├── README.md
└── report.pdf                          
```

Files created by the notebook (saved to Google Drive in `MyDrive/nlp_project/`):

| File | Content |
|---|---|
| `bpe.model`, `bpe.vocab` | trained tokenizer |
| `train.csv`, `val.csv`, `test.csv` | the exact data splits |
| `baseline_best.pt`, `attention_best.pt` | weights of the best validation epoch |
| `baseline_last.pt`, `attention_last.pt` | last checkpoint (used to resume training) |
| `*_history.json` | train/validation loss per epoch and training time |
| `results.csv`, `test_generations.csv` | metrics table and all generated test replies |
| `learning_curves.png`, `comparison.png` | figures used in the report |

## 8. How to run

1. Open `NLP_Project_Customer_support_response_generation.ipynb` in **Google Colab** and select a GPU (Runtime → Change runtime type).
2. Run the cells from top to bottom. The first cell installs the dependencies and mounts Google Drive:
   ```
   pip install datasets sentencepiece sacrebleu rouge-score
   ```
   Other libraries used: `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`.
3. Training takes about 42 min (baseline) and 92 min (attention) for 20 epochs on a Colab GPU. It is **resumable**: if Colab disconnects, re-run the definition cells and the training cell, and it continues from the last finished epoch.
4. To evaluate without retraining, reload the tokenizer, the CSV splits and the `*_best.pt` weights from Drive, then run the evaluation cells.

Random seed: 42.

## 9. Notebook structure (follows the project description)

| Notebook section | Content |
|---|---|
| 1–2 | Dataset loading and description, cleaning, train/validation/test split |
| 3–4 | Subword vocabulary, special tokens, padding, truncation, DataLoaders |
| 5–7 | Encoder, decoder, attention, teacher forcing |
| 8 | Training with progress bar, learning curves |
| 9 | Greedy generation |
| 10 | Evaluation: perplexity, token accuracy, BLEU, ROUGE |
| 11 | Baseline vs. attention comparison |
| 12 | Best and worst examples, failure analysis |

## 10. Limitations

- The dataset is synthetic and repetitive, so scores are likely higher than on real customer conversations.
- Each model was trained once (one seed), so small differences may partly be noise.
- Responses longer than 256 tokens (4.4%) are truncated.
- BLEU and ROUGE use a single reference and penalize valid paraphrases.
- Only greedy decoding was tested.
- The models learn the style of support replies, not real company facts, so they can invent details.
