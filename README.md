# Research Progress

Notes, notebooks, and code tracking my progress during my research assistantship.

## Progress log

| Date | Topic | Notebook |
|---|---|---|
| 2026-10-11 | Additive (Bahdanau) attention from scratch in NumPy and PyTorch | [additive_attention_explained.ipynb](notebooks/additive_attention_explained.ipynb) |
| 2026-10-25 | IMDB sentiment analysis: recurrent (GRU) vs. recurrent + additive attention, with DataLoader and training loop | [imdb_rnn_vs_attention.ipynb](notebooks/imdb_rnn_vs_attention.ipynb) |
| 2026-11-02 | Addressed Oct 27 feedback: bag-of-words + logistic regression baseline, bidirectional GRU + attention, transformer encoder built from its building blocks; LR 3e-4 for 25 epochs, half-precision and DataLoader timing | [imdb_baselines_bigru_transformer.ipynb](notebooks/imdb_baselines_bigru_transformer.ipynb) |
| 2026-11-09 | Fine-tuning pretrained transformers on IMDB: DistilBERT with 0, 2, 4 or all layers unfrozen, then BERT-base and RoBERTa-base with all layers; head+tail truncation to 510 tokens; RoBERTa reaches 95.7% test accuracy | [imdb_finetune_pretrained.ipynb](notebooks/imdb_finetune_pretrained.ipynb) |
| 2026-11-09 | German→English translation on Multi30k: GRU encoder-decoder with additive attention vs. a transformer built from its building blocks, shared SentencePiece vocabulary, greedy and beam search, scored with BLEU | [translation_multi30k.ipynb](notebooks/translation_multi30k.ipynb) |
