"""Alternating Least Squares (ALS) Matrix Factorization with EMA Sufficient Statistics (Track T06)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np

from core.schedule import CosineSchedule, ConstantSchedule, StepSchedule


@dataclass
class EMAAlternatingLeastSquares:
    """Mini-batch ALS recommender model updating item statistics via EMA with cosine annealing."""

    n_users: int
    n_items: int
    n_factors: int = 16
    reg: float = 0.05
    alpha0: float = 0.6
    total_steps: int = 100
    eps: float = 0.05
    seed: int = 42

    user_factors: np.ndarray = field(init=False)
    item_factors: np.ndarray = field(init=False)
    item_A: np.ndarray = field(init=False)  # (n_items, f, f) running outer products
    item_b: np.ndarray = field(init=False)  # (n_items, f) running linear projections
    schedule: CosineSchedule = field(init=False)
    step: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.n_factors)
        self.user_factors = rng.normal(0.0, scale, size=(self.n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0.0, scale, size=(self.n_items, self.n_factors)).astype(np.float32)

        # Initialize item statistics with prior regularization
        eye = np.eye(self.n_factors, dtype=np.float32) * self.reg
        self.item_A = np.tile(eye[None, :, :], (self.n_items, 1, 1))
        self.item_b = np.zeros((self.n_items, self.n_factors), dtype=np.float32)

        self.schedule = CosineSchedule(alpha0=self.alpha0, total_steps=self.total_steps, eps=self.eps)
        self.step = 0

    def initialize_statistics(self, user_ratings: dict[int, list[tuple[int, float]]]) -> None:
        """Warm-start sufficient statistics from initial factor projections."""
        f = self.n_factors
        self.item_A = np.zeros((self.n_items, f, f), dtype=np.float32)
        self.item_b = np.zeros((self.n_items, f), dtype=np.float32)
        for u in range(self.n_users):
            u_vec = self.user_factors[u]
            u_outer = np.outer(u_vec, u_vec)
            for item_id, r in user_ratings.get(u, []):
                self.item_A[item_id] += u_outer
                self.item_b[item_id] += r * u_vec

    def update_user_batch(
        self,
        batch_users: np.ndarray,
        user_ratings: dict[int, list[tuple[int, float]]],
    ) -> float:
        """Process one mini-batch of users, update user factors, EMA update item statistics."""
        batch_size = len(batch_users)
        if batch_size == 0:
            return 0.0

        f = self.n_factors
        reg_eye = np.eye(f, dtype=np.float32) * self.reg
        alpha = self.schedule(self.step)

        batch_squared_errors = []

        # 1. Update user factors for users in this batch
        for u in batch_users:
            ratings = user_ratings.get(int(u), [])
            if not ratings:
                continue

            item_indices = [item_id for item_id, _ in ratings]
            r_vals = np.array([r for _, r in ratings], dtype=np.float32)

            V_u = self.item_factors[item_indices]  # (len(ratings), f)
            A_u = V_u.T @ V_u + reg_eye
            b_u = V_u.T @ r_vals

            u_vec = np.linalg.solve(A_u, b_u)
            self.user_factors[u] = u_vec

            preds = V_u @ u_vec
            batch_squared_errors.extend(list((preds - r_vals) ** 2))

            # Accumulate statistics and update item factors
            u_outer = np.outer(u_vec, u_vec)
            for item_id, r in ratings:
                self.item_A[item_id] = (1.0 - alpha) * self.item_A[item_id] + alpha * u_outer
                self.item_b[item_id] = (1.0 - alpha) * self.item_b[item_id] + alpha * r * u_vec
                A_solve = self.item_A[item_id] + reg_eye
                self.item_factors[item_id] = np.linalg.solve(A_solve, self.item_b[item_id])

        self.step += 1
        return float(np.mean(batch_squared_errors)) if batch_squared_errors else 0.0

    def compute_rmse(self, test_triplets: list[tuple[int, int, float]]) -> float:
        """Compute RMSE on held-out user-item-rating test interactions."""
        if not test_triplets:
            return 0.0

        u_idx = np.array([t[0] for t in test_triplets], dtype=np.int64)
        i_idx = np.array([t[1] for t in test_triplets], dtype=np.int64)
        r_true = np.array([t[2] for t in test_triplets], dtype=np.float32)

        preds = np.sum(self.user_factors[u_idx] * self.item_factors[i_idx], axis=1)
        mse = float(np.mean((preds - r_true) ** 2))
        return float(np.sqrt(mse))


@dataclass
class FullBatchALS:
    """Classic full-batch Alternating Least Squares (alternates all users then all items)."""

    n_users: int
    n_items: int
    n_factors: int = 16
    reg: float = 0.05
    seed: int = 42

    user_factors: np.ndarray = field(init=False)
    item_factors: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.n_factors)
        self.user_factors = rng.normal(0.0, scale, size=(self.n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0.0, scale, size=(self.n_items, self.n_factors)).astype(np.float32)

    def run_epoch(
        self,
        user_ratings: dict[int, list[tuple[int, float]]],
        item_ratings: dict[int, list[tuple[int, float]]],
    ) -> float:
        """Run one full pass updating all users then all items."""
        f = self.n_factors
        reg_eye = np.eye(f, dtype=np.float32) * self.reg

        # Update all users
        for u in range(self.n_users):
            ratings = user_ratings.get(u, [])
            if not ratings:
                continue
            item_indices = [item_id for item_id, _ in ratings]
            r_vals = np.array([r for _, r in ratings], dtype=np.float32)
            V_u = self.item_factors[item_indices]
            A_u = V_u.T @ V_u + reg_eye
            b_u = V_u.T @ r_vals
            self.user_factors[u] = np.linalg.solve(A_u, b_u)

        # Update all items
        for i in range(self.n_items):
            ratings = item_ratings.get(i, [])
            if not ratings:
                continue
            user_indices = [user_id for user_id, _ in ratings]
            r_vals = np.array([r for _, r in ratings], dtype=np.float32)
            U_i = self.user_factors[user_indices]
            A_i = U_i.T @ U_i + reg_eye
            b_i = U_i.T @ r_vals
            self.item_factors[i] = np.linalg.solve(A_i, b_i)

        return 0.0

    def compute_rmse(self, test_triplets: list[tuple[int, int, float]]) -> float:
        u_idx = np.array([t[0] for t in test_triplets], dtype=np.int64)
        i_idx = np.array([t[1] for t in test_triplets], dtype=np.int64)
        r_true = np.array([t[2] for t in test_triplets], dtype=np.float32)
        preds = np.sum(self.user_factors[u_idx] * self.item_factors[i_idx], axis=1)
        mse = float(np.mean((preds - r_true) ** 2))
        return float(np.sqrt(mse))
