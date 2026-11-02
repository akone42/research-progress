# Experiment log: imdb_baselines_bigru_transformer.ipynb

A running record of what was run overnight, what changed, and why. Times are local (EST).

## 2026-11-01

**Benchmarks before writing the notebook (M2 Pro, PyTorch 2.11)**
- Training step at batch 64, ~230 tokens:
  - one-direction GRU: CPU 297 ms, MPS 296 ms
  - BiGRU: CPU 565 ms, MPS 569 ms
  - transformer: CPU 339 ms, MPS 57 ms
  → Use MPS for everything.
- Half precision (autocast) was slower everywhere on this Mac:
  - BiGRU on MPS: 513 → ~870 ms
  - transformer on CPU with bf16: 339 → 2212 ms
  → Train in fp32. The notebook re-measures this in section 8.
- Review lengths (train), tokens:
  - median 174, 95th percentile 598, 99th percentile 911, max 2473
  - MAX_LEN 256 keeps 76.6% of tokens; 1024 keeps 99.8%. Chose 1024.

**23:07 smoke test** (1k train / 500 val / 500 test, 2 epochs): every cell ran and every check passed. Notes:
- Padding share: random batches 73%, length-bucketed batches 9%.
- DataLoader workers: 2 epochs of batches took 0.01 s with `num_workers=0`, 1.5 s with 2 workers, 1.2 s with 4. The data is already encoded in memory, so workers only add process start-up and copying cost. Use 0.

**23:11 full run started.** Settings:
- LR 3e-4, 25 epochs, batch 64, MAX_LEN 1024, fp32 on MPS
- models: BoW+LR, GRU+attention (one direction), BiGRU+attention, transformer (2 layers, 4 heads, d=128)

## 2026-11-02

**~00:20. Run 1 crashed.** One-direction GRU + attention trained all 25 epochs, but its results were lost: `nbconvert` only writes the notebook at the end. BiGRU epoch 1 took 434 s, about twice the estimate. Epoch 2 then failed with `MPS backend out of memory`: PyTorch tensors were using 0.68 GB, while the driver's "other allocations" had grown to 19.46 GB.

**Diagnosis**
- Isolated BiGRU test on MPS: driver memory grew 1.67 → 2.70 GB over 60 steps of the *same* shape. That's a leak in the MPS GRU path, not caused by bucketing.
- The notebook's transformer on MPS, with realistic varying batch lengths, also pushed the machine into swapping (very low CPU use, ~4k free memory pages). Test stopped.

**Decision:** train every model on the **CPU**.
- The benchmark showed CPU and MPS tie for the GRUs, so they lose nothing.
- The transformer loses its ~6× MPS speed-up and costs about 35 extra minutes. That's worth it for a run that won't crash.
- The notebook's setup section now explains this.

**Other change:** the notebook now runs through a small runner (`nbclient`) that saves it after every cell. If something fails again, finished models keep their outputs.

**Smoke test on CPU** (1k reviews, 2 epochs): passed. Estimated full-run time: GRU ~57 min, BiGRU ~106 min, transformer ~36 min.

**00:40. Run 2 started** (CPU). All other settings unchanged.

**08:20. Run 2 finished** after 471.7 min. All cells ran, no errors. The run took over twice the estimate: several epochs took 1000–1450 s instead of ~200–500 s, scattered across all three models. The Mac was probably throttled or idling the process overnight despite `caffeinate -i`.

| model | test acc | test loss | best epoch (by val acc) | epoch with lowest val loss | train time |
|---|---|---|---|---|---|
| bag-of-words + LR (C=0.1) | **0.8805** | 0.294 | - | - | 4 s |
| GRU + attention | 0.8718 | 0.360 | 9 | 6 (0.306) | 145.7 min |
| BiGRU + attention | 0.8662 | 1.042 | 25 | 5 (0.305) | 252.0 min |
| transformer | 0.8571 | 0.631 | 24 | 9 (0.335) | 65.2 min |

Other measurements:
- Half precision on CPU, ms per batch:
  - BiGRU: fp32 1343, fp16 2463, bf16 2829
  - transformer: fp32 370, fp16 4022, bf16 4049
- DataLoader, time to iterate 2 epochs: 0.11 s with 0 workers, 4.1 s with 2, 6.4 s with 4.

**Decision: no retrain.** I considered rerunning with checkpoints chosen by validation loss plus early stopping.
- It would take ~2.5 h even with early stopping (more at last night's speeds).
- It wouldn't change the conclusions: accuracy at the lowest-val-loss epochs is about the same (0.866–0.874 val).
- The flaw is written up in the notebook (results section, point 5) as the first next step.
- The 25-epoch curves are what McNeill asked for, and they show the overfitting clearly.

**Notebook edits after the run** (markdown and display only, no code that affects results):
- Filled in the DataLoader-workers, half-precision and results/conclusions notes.
- Length-bucket table: cast the review counts to int (they showed as 3,163.0). The output HTML was patched to match.

**Files**
- `notebooks/imdb_baselines_bigru_transformer.ipynb`: executed, with outputs.
- `notebooks/imdb_data.py`: Dataset, collate function and length-bucketing sampler.
- `notebooks/experiment_log.md`: this file.
- README progress-log row added. Nothing committed.
