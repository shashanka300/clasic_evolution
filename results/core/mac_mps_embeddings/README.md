T00 acceptance as run on the Mac (Apple M1 Pro, MPS), on embeddings computed there. Kept for the record; not read by
the evaluator. T00 was re-run on the RTX 5090 desktop (`core_t00` in results/jobs_5090.txt) because the desktop's
data/ was re-downloaded and its AG News / DINOv2 embeddings were recomputed on CUDA, which differ from the MPS ones in
the last bits; the re-run puts T00 on the same embedding version as every other track. Mac verdict: SURVIVES.
