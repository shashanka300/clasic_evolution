Produced by commits before the lda_v2 E-step aliasing fix (2026-11-14). With `numba.njit(parallel=True)` the per-iteration
`newg = np.empty(K)` was hoisted out of the loop, so `g = newg` aliased the two buffers, the convergence test read
`diff = 0` and every document's variational update stopped after 2 iterations (instead of running to tol=1e-3 /
max_iter). Every method (batch VB, online LDA, cosine EMA, Hoffman default, perplexity evaluation) shared that
truncated E-step. Regression test: `tests/test_gpu_ports.py::test_lda_estep_iterates_to_tol`. Superseded; kept for
the record. Not read by the evaluator.
