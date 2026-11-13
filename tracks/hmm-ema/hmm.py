"""Online Baum-Welch for Hidden Markov Models with EMA Sufficient Statistics and State Revival (Track T07)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np
import scipy.special as sp

from core.schedule import CosineSchedule, ConstantSchedule, StepSchedule


def log_normalize(log_p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize log probabilities along axis using logsumexp."""
    log_sum = sp.logsumexp(log_p, axis=axis, keepdims=True)
    return log_p - log_sum


@dataclass
class EMABaumWelch:
    """Mini-batch Baum-Welch with EMA running counts and cosine annealing schedule."""

    n_states: int
    n_emissions: int
    n_seqs_total: int
    alpha0: float = 0.5
    total_steps: int = 100
    eps: float = 0.05
    revival_interval: int = 5
    revival_threshold: float = 0.15
    seed: int = 42

    log_pi: np.ndarray = field(init=False)
    log_A: np.ndarray = field(init=False)
    log_B: np.ndarray = field(init=False)
    counts_A: np.ndarray = field(init=False)
    counts_B: np.ndarray = field(init=False)
    schedule: CosineSchedule = field(init=False)
    step: int = 0
    revivals: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # Random Dirichlet initialization of HMM parameters
        pi_init = rng.dirichlet(np.ones(self.n_states))
        A_init = rng.dirichlet(np.ones(self.n_states) * 2.0, size=self.n_states)
        B_init = rng.dirichlet(np.ones(self.n_emissions) * 0.5, size=self.n_states)

        self.log_pi = np.log(pi_init + 1e-12)
        self.log_A = np.log(A_init + 1e-12)
        self.log_B = np.log(B_init + 1e-12)

        # Initial expected counts
        self.counts_A = (A_init * 100.0).astype(np.float32)
        self.counts_B = (B_init * 100.0).astype(np.float32)

        self.schedule = CosineSchedule(alpha0=self.alpha0, total_steps=self.total_steps, eps=self.eps)
        self.step = 0
        self.revivals = 0

    def _forward_backward(self, sequence: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute forward and backward variables in log space for a discrete sequence.

        Returns (log_lik, gamma, xi) where:
        gamma is (T, S) posterior state probabilities,
        xi is (T-1, S, S) posterior transition probabilities.
        """
        T = len(sequence)
        S = self.n_states

        log_alpha = np.zeros((T, S), dtype=np.float32)
        log_alpha[0] = self.log_pi + self.log_B[:, sequence[0]]

        # Forward pass
        for t in range(1, T):
            for j in range(S):
                log_alpha[t, j] = sp.logsumexp(log_alpha[t - 1] + self.log_A[:, j]) + self.log_B[j, sequence[t]]

        log_lik = float(sp.logsumexp(log_alpha[T - 1]))

        # Backward pass
        log_beta = np.zeros((T, S), dtype=np.float32)
        log_beta[T - 1] = 0.0

        for t in range(T - 2, -1, -1):
            for i in range(S):
                log_beta[t, i] = sp.logsumexp(self.log_A[i, :] + self.log_B[:, sequence[t + 1]] + log_beta[t + 1, :])

        # Posterior state probabilities gamma
        log_gamma = log_alpha + log_beta - log_lik
        gamma = np.exp(log_gamma)

        # Posterior transitions xi
        xi = np.zeros((T - 1, S, S), dtype=np.float32)
        for t in range(T - 1):
            log_xi_t = (
                log_alpha[t, :, None]
                + self.log_A
                + self.log_B[None, :, sequence[t + 1]]
                + log_beta[t + 1, None, :]
                - log_lik
            )
            xi[t] = np.exp(log_xi_t)

        return log_lik, gamma, xi

    def update_batch(self, sequences: list[np.ndarray]) -> float:
        """Process one mini-batch of sequences, update running counts with EMA, update parameters."""
        batch_size = len(sequences)
        if batch_size == 0:
            return 0.0

        S = self.n_states
        V = self.n_emissions

        batch_A = np.zeros((S, S), dtype=np.float32)
        batch_B = np.zeros((S, V), dtype=np.float32)
        total_log_lik = 0.0

        for seq in sequences:
            if len(seq) < 2:
                continue
            log_lik, gamma, xi = self._forward_backward(seq)
            total_log_lik += log_lik

            # Accumulate transitions
            batch_A += np.sum(xi, axis=0)

            # Accumulate emissions
            for t, token in enumerate(seq):
                batch_B[:, int(token)] += gamma[t]

        # EMA update with schedule
        alpha = self.schedule(self.step)
        scale = float(self.n_seqs_total) / float(batch_size)

        self.counts_A = (1.0 - alpha) * self.counts_A + alpha * scale * batch_A
        self.counts_B = (1.0 - alpha) * self.counts_B + alpha * scale * batch_B

        # Normalization with Laplace smoothing
        A_norm = (self.counts_A + 1e-4) / np.sum(self.counts_A + 1e-4, axis=1, keepdims=True)
        B_norm = (self.counts_B + 1e-4) / np.sum(self.counts_B + 1e-4, axis=1, keepdims=True)

        self.log_A = np.log(A_norm)
        self.log_B = np.log(B_norm)

        # State revival every N batches
        if self.step > 0 and self.step % self.revival_interval == 0:
            self._revive_unused_states()

        self.step += 1
        return total_log_lik / batch_size

    def _revive_unused_states(self) -> None:
        """Detect starved states with near-zero emission counts and revive them by splitting."""
        state_masses = np.sum(self.counts_B, axis=1)
        mean_mass = float(np.mean(state_masses))
        threshold = self.revival_threshold * mean_mass

        dead_idx = np.flatnonzero(state_masses < threshold)
        if len(dead_idx) == 0:
            return

        live_idx = np.flatnonzero(state_masses >= threshold)
        if len(live_idx) == 0:
            return

        donor_i = live_idx[np.argmax(state_masses[live_idx])]
        donor_vec = self.counts_B[donor_i].copy()

        d = self.n_emissions
        sign = np.where(np.arange(d) % 2 == 0, 1.0, -1.0).astype(np.float32)
        delta = 0.05

        for dead_i in dead_idx:
            half = 0.5 * donor_vec
            self.counts_B[donor_i] = half * (1.0 - delta * sign)
            self.counts_B[dead_i] = half * (1.0 + delta * sign)
            self.revivals += 1

        B_norm = (self.counts_B + 1e-4) / np.sum(self.counts_B + 1e-4, axis=1, keepdims=True)
        self.log_B = np.log(B_norm)

    def evaluate_log_likelihood(self, test_sequences: list[np.ndarray]) -> float:
        """Compute average negative log-likelihood per token on test sequences."""
        total_ll = 0.0
        total_tokens = 0
        for seq in test_sequences:
            if len(seq) < 2:
                continue
            log_lik, _, _ = self._forward_backward(seq)
            total_ll += log_lik
            total_tokens += len(seq)
        if total_tokens == 0:
            return 0.0
        return float(-total_ll / total_tokens)


@dataclass
class FullBatchBaumWelch:
    """Standard full-batch Baum-Welch HMM optimizer."""

    n_states: int
    n_emissions: int
    seed: int = 42

    log_pi: np.ndarray = field(init=False)
    log_A: np.ndarray = field(init=False)
    log_B: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        pi_init = rng.dirichlet(np.ones(self.n_states))
        A_init = rng.dirichlet(np.ones(self.n_states) * 2.0, size=self.n_states)
        B_init = rng.dirichlet(np.ones(self.n_emissions) * 0.5, size=self.n_states)
        self.log_pi = np.log(pi_init + 1e-12)
        self.log_A = np.log(A_init + 1e-12)
        self.log_B = np.log(B_init + 1e-12)

    def run_epoch(self, sequences: list[np.ndarray]) -> float:
        """Run one full E-M pass over all sequences."""
        S = self.n_states
        V = self.n_emissions
        counts_A = np.zeros((S, S), dtype=np.float32)
        counts_B = np.zeros((S, V), dtype=np.float32)

        for seq in sequences:
            if len(seq) < 2:
                continue
            # Forward pass
            T = len(seq)
            log_alpha = np.zeros((T, S), dtype=np.float32)
            log_alpha[0] = self.log_pi + self.log_B[:, seq[0]]
            for t in range(1, T):
                for j in range(S):
                    log_alpha[t, j] = sp.logsumexp(log_alpha[t - 1] + self.log_A[:, j]) + self.log_B[j, seq[t]]
            log_lik = float(sp.logsumexp(log_alpha[T - 1]))

            # Backward pass
            log_beta = np.zeros((T, S), dtype=np.float32)
            for t in range(T - 2, -1, -1):
                for i in range(S):
                    log_beta[t, i] = sp.logsumexp(self.log_A[i, :] + self.log_B[:, seq[t + 1]] + log_beta[t + 1, :])

            log_gamma = log_alpha + log_beta - log_lik
            gamma = np.exp(log_gamma)

            for t in range(T - 1):
                log_xi_t = (
                    log_alpha[t, :, None]
                    + self.log_A
                    + self.log_B[None, :, seq[t + 1]]
                    + log_beta[t + 1, None, :]
                    - log_lik
                )
                counts_A += np.exp(log_xi_t)

            for t, token in enumerate(seq):
                counts_B[:, int(token)] += gamma[t]

        A_norm = (counts_A + 1e-4) / np.sum(counts_A + 1e-4, axis=1, keepdims=True)
        B_norm = (counts_B + 1e-4) / np.sum(counts_B + 1e-4, axis=1, keepdims=True)
        self.log_A = np.log(A_norm)
        self.log_B = np.log(B_norm)
        return 0.0

    def evaluate_log_likelihood(self, test_sequences: list[np.ndarray]) -> float:
        total_ll = 0.0
        total_tokens = 0
        S = self.n_states
        for seq in test_sequences:
            if len(seq) < 2:
                continue
            T = len(seq)
            log_alpha = np.zeros((T, S), dtype=np.float32)
            log_alpha[0] = self.log_pi + self.log_B[:, seq[0]]
            for t in range(1, T):
                for j in range(S):
                    log_alpha[t, j] = sp.logsumexp(log_alpha[t - 1] + self.log_A[:, j]) + self.log_B[j, seq[t]]
            total_ll += float(sp.logsumexp(log_alpha[T - 1]))
            total_tokens += T
        return float(-total_ll / max(total_tokens, 1))

