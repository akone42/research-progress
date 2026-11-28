# Research Progress

Notes, notebooks, and code from my research assistantship with Prof. Matthew McNeill.

## Current project: predicting the next turn's speech features

- **Question:** Can a transformer predict the speech features of the next turn in a conversation (pitch, loudness, voice quality, and speaking rate) better than a recurrent network?
- **Setup:** Telephone conversations from the Fisher corpus. A causal transformer vs. a GRU and a GRU with attention, each tuned on size and learning rate, compared with paired bootstrap confidence intervals over test conversations. The corpus is licensed, so the data is not in this repo.
- **Result so far:** Not on this task. The tuned GRU with attention (test MSE 0.7737) edges out a 4-layer transformer (0.7760) with about 18× fewer parameters, and all three models remove about 24% of the error of predicting the mean. The gaps are small, so the ordering is suggestive rather than settled. The transformer does best when the speaker changes and worst when the same speaker continues, and adding the words gives a small but measurable gain for the models with attention.
- **Notebook:** [fisher_next_turn_prediction.ipynb](notebooks/fisher_next_turn_prediction.ipynb)

## Progress log (newest first)

| Date | Topic | Notebook |
|---|---|---|
| 2026-11-27 | Follow-up on Fisher: under what conditions can a transformer outperform a recurrent network? ALiBi recency bias and a learned speaker bias, 3 seeds with bootstrap confidence intervals. ALiBi brings the transformer level with GRU + attention, and regularization doesn't change that. When the models are told who speaks next, transformer + ALiBi beats GRU + attention | [fisher_transformer_attention_bias.ipynb](notebooks/fisher_transformer_attention_bias.ipynb) |
| 2026-11-26 | Predicting the next conversational turn's 7 speech features on the Fisher corpus: causal transformer vs GRU and GRU + attention, with a size and learning-rate sweep. The tuned transformer beats the GRU but not GRU + attention. Extension: words + speech features and words only | [fisher_next_turn_prediction.ipynb](notebooks/fisher_next_turn_prediction.ipynb) |
| 2026-11-09 | German→English translation on Multi30k: GRU encoder-decoder with additive attention vs. a transformer built from its building blocks, shared SentencePiece vocabulary, greedy and beam search, scored with BLEU | [translation_multi30k.ipynb](notebooks/translation_multi30k.ipynb) |
| 2026-11-09 | Fine-tuning pretrained transformers on IMDB: DistilBERT with 0, 2, 4 or all layers unfrozen, then BERT-base and RoBERTa-base with all layers; head+tail truncation to 510 tokens; RoBERTa reaches 95.7% test accuracy | [imdb_finetune_pretrained.ipynb](notebooks/imdb_finetune_pretrained.ipynb) |
| 2026-11-02 | Addressed Oct 27 feedback: bag-of-words + logistic regression baseline, bidirectional GRU + attention, transformer encoder built from its building blocks; LR 3e-4 for 25 epochs, half-precision and DataLoader timing | [imdb_baselines_bigru_transformer.ipynb](notebooks/imdb_baselines_bigru_transformer.ipynb) |
| 2026-10-25 | IMDB sentiment analysis: recurrent (GRU) vs. recurrent + additive attention, with DataLoader and training loop | [imdb_rnn_vs_attention.ipynb](notebooks/imdb_rnn_vs_attention.ipynb) |
| 2026-10-11 | Additive (Bahdanau) attention from scratch in NumPy and PyTorch | [additive_attention_explained.ipynb](notebooks/additive_attention_explained.ipynb) |
