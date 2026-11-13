"""Online Variational Bayes for LDA with Cosine Schedules and Scaling Laws (Track T05)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np
import scipy.special as sp

from core.schedule import CosineSchedule, InverseTimeSchedule, StepSchedule


def dirichlet_expectation(alpha: np.ndarray) -> np.ndarray:
    """Compute E[log theta] for Dirichlet(alpha) using digamma function."""
    if alpha.ndim == 1:
        return sp.digamma(alpha) - sp.digamma(np.sum(alpha))
    return sp.digamma(alpha) - sp.digamma(np.sum(alpha, axis=-1, keepdims=True))


def predict_lda_alpha0(n_docs: int, n_topics: int, vocab_size: int, batch_size: int) -> float:
    """Predict optimal initial step size alpha0 for online LDA via scaling law."""
    ratio = max(n_docs / max(batch_size, 1), 1.0)
    # Scaling law: higher dataset-to-batch ratio anneals smoother with moderate alpha0
    alpha0 = 0.8 * (ratio ** -0.15) * (n_topics ** -0.05)
    return float(np.clip(alpha0, 0.05, 0.95))


@dataclass
class OnlineLDA:
    """Online Latent Dirichlet Allocation with configurable step schedule and topic revival."""

    n_topics: int
    vocab_size: int
    n_docs_total: int
    schedule_type: str = "cosine"  # "cosine" or "hoffman"
    alpha0: Optional[float] = None
    total_steps: int = 100
    tau0: float = 64.0
    kappa: float = 0.7
    alpha_prior: float = 0.1  # Document-topic Dirichlet prior
    eta_prior: float = 0.01  # Topic-word Dirichlet prior
    revival_interval: int = 5
    revival_threshold: float = 0.15
    seed: int = 42

    lambda_mat: np.ndarray = field(init=False)  # Variational topic-word params (k, vocab)
    schedule: StepSchedule = field(init=False)
    step: int = 0
    revivals: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # Initialize variational topic-word parameters around eta
        self.lambda_mat = rng.gamma(100.0, 1.0 / 100.0, size=(self.n_topics, self.vocab_size)).astype(np.float32)
        self.lambda_mat += self.eta_prior

        if self.alpha0 is None:
            self.alpha0 = predict_lda_alpha0(self.n_docs_total, self.n_topics, self.vocab_size, batch_size=64)

        if self.schedule_type == "cosine":
            self.schedule = CosineSchedule(alpha0=self.alpha0, total_steps=self.total_steps, eps=0.05)
        else:
            # Hoffman (2010) default schedule: (tau0 + t)^(-kappa)
            self.schedule = InverseTimeSchedule(alpha0=1.0, tau0=self.tau0, kappa=self.kappa)

        self.step = 0
        self.revivals = 0

    def _e_step(
        self,
        doc_word_counts: List[dict[int, float]],
        max_iter: int = 25,
        tol: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Variational E-step for a mini-batch of documents.

        Returns (gamma, sstats) where gamma is (batch_size, n_topics)
        and sstats is (n_topics, vocab_size).
        """
        batch_size = len(doc_word_counts)
        # E[log beta]
        e_log_beta = dirichlet_expectation(self.lambda_mat)  # (k, V)

        gamma = np.full((batch_size, self.n_topics), self.alpha_prior + self.vocab_size / self.n_topics, dtype=np.float32)
        sstats = np.zeros_like(self.lambda_mat, dtype=np.float32)

        for d_idx, word_counts in enumerate(doc_word_counts):
            if not word_counts:
                continue
            words = np.array(list(word_counts.keys()), dtype=np.int64)
            counts = np.array(list(word_counts.values()), dtype=np.float32)

            g_d = gamma[d_idx]
            for _ in range(max_iter):
                old_g_d = g_d.copy()
                e_log_theta_d = dirichlet_expectation(g_d)  # (k,)

                # phi matrix: (len(words), k)
                phi = np.exp(e_log_theta_d[None, :] + e_log_beta[:, words].T)
                phi /= np.maximum(np.sum(phi, axis=1, keepdims=True), 1e-12)

                g_d = self.alpha_prior + np.sum(counts[:, None] * phi, axis=0)
                if np.max(np.abs(g_d - old_g_d)) < tol:
                    break

            gamma[d_idx] = g_d
            # Accumulate sufficient statistics
            e_log_theta_d = dirichlet_expectation(g_d)
            phi = np.exp(e_log_theta_d[None, :] + e_log_beta[:, words].T)
            phi /= np.maximum(np.sum(phi, axis=1, keepdims=True), 1e-12)
            sstats[:, words] += (counts[:, None] * phi).T

        return gamma, sstats

    def update_batch(self, doc_word_counts: List[dict[int, float]]) -> float:
        """Process one mini-batch, update variational parameters via EMA schedule."""
        batch_size = max(len(doc_word_counts), 1)
        gamma, sstats = self._e_step(doc_word_counts)

        # Scale sufficient statistics up to whole dataset (Hoffman eq 5)
        scale = float(self.n_docs_total) / float(batch_size)
        lambda_tilde = self.eta_prior + scale * sstats

        rho = self.schedule(self.step)
        # EMA update
        self.lambda_mat = (1.0 - rho) * self.lambda_mat + rho * lambda_tilde

        # Periodic dead topic revival
        if self.step > 0 and self.step % self.revival_interval == 0:
            self._revive_dead_topics()

        self.step += 1
        return float(np.mean(gamma))

    def _revive_dead_topics(self) -> None:
        """Detect under-utilized topics and revive them by donor splitting."""
        topic_masses = np.sum(self.lambda_mat, axis=1)
        mean_mass = float(np.mean(topic_masses))
        threshold = self.revival_threshold * mean_mass

        dead_idx = np.flatnonzero(topic_masses < threshold)
        if len(dead_idx) == 0:
            return

        live_idx = np.flatnonzero(topic_masses >= threshold)
        if len(live_idx) == 0:
            return

        donor_idx = live_idx[np.argmax(topic_masses[live_idx])]
        delta = 0.05
        d = self.vocab_size
        sign = np.where(np.arange(d) % 2 == 0, 1.0, -1.0).astype(np.float32)

        donor_vec = self.lambda_mat[donor_idx]
        half_donor = 0.5 * donor_vec

        for dead_i in dead_idx:
            self.lambda_mat[donor_idx] = half_donor * (1.0 - delta * sign)
            self.lambda_mat[dead_i] = half_donor * (1.0 + delta * sign)
            self.revivals += 1

    def compute_perplexity(self, doc_word_counts: List[dict[int, float]]) -> float:
        """Compute held-out perplexity on evaluation documents."""
        if not doc_word_counts:
            return 1.0

        e_log_beta = dirichlet_expectation(self.lambda_mat)
        # Normalized topic-word distributions
        beta = np.exp(e_log_beta)
        beta /= np.maximum(np.sum(beta, axis=1, keepdims=True), 1e-12)

        gamma, _ = self._e_step(doc_word_counts, max_iter=15)
        # Expected topic mixture for each doc
        theta = gamma / np.maximum(np.sum(gamma, axis=1, keepdims=True), 1e-12)

        log_per_word = 0.0
        total_words = 0

        for d_idx, word_counts in enumerate(doc_word_counts):
            if not word_counts:
                continue
            words = list(word_counts.keys())
            counts = list(word_counts.values())

            # p(w | doc) = sum_k theta_{d, k} * beta_{k, w}
            doc_probs = theta[d_idx] @ beta[:, words]
            doc_log_lik = np.sum(np.array(counts) * np.log(np.maximum(doc_probs, 1e-12)))

            log_per_word += float(doc_log_lik)
            total_words += sum(counts)

        if total_words == 0:
            return 1.0
        return float(np.exp(-log_per_word / max(total_words, 1)))

