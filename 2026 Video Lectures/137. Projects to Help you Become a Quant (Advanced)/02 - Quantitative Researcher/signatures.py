"""
signatures.py
=============

**Path signatures** and a small, honest illustration of **model-free /
data-driven pricing** with them.

What is a signature?
--------------------
For a (piecewise-smooth) path ``X : [0, T] -> R^d`` the *signature* is the
infinite collection of iterated integrals

    S(X) = ( 1,  {int dX^i},  {int int dX^i dX^j},  {int int int dX^i dX^j dX^k}, ... )

living in the tensor algebra. Truncating at level ``M`` gives a *finite* feature
vector of dimension ``1 + d + d^2 + ... + d^M = (d^{M+1} - 1) / (d - 1)``. Two
facts make it a superb feature map (Lyons; Kidger & Lyons):

  * **Universality.** Any continuous functional of the path can be approximated
    arbitrarily well by a *linear* functional of a high-enough signature. So a
    complicated, path-dependent payoff ``F(X)`` is approximately ``<l, S(X)>``
    for some coefficient tensor ``l``.
  * **Chen's identity.** The signature of a concatenation of paths is the tensor
    product of their signatures, which is exactly how we compute it below: split
    the path into straight segments, take the tensor *exponential* of each
    increment, and Chen-multiply them together.

Model-free pricing philosophy
------------------------------
Because the payoff is (approximately) *linear* in the signature,

    price = E[ F(X) ] ~ E[ <l, S(X)> ] = < l, E[S(X)] >.

The pricing problem factorises into (i) a payoff object ``l`` -- learnable once
from data or examples -- and (ii) the **expected signature** ``E[S(X)]``, a
property of the *market* / measure. Given the expected signature, the *same*
``l`` prices the payoff under *any* dynamics: no SDE, no calibration of a
specific model. Here we illustrate this by learning ``l`` for an arithmetic
Asian option via linear regression on simulated paths, then pricing it as a
linear functional of the expected signature -- and checking it against plain
Monte-Carlo.

Dependencies
------------
An optional C-accelerated ``iisignature`` package is auto-detected, but a pure
NumPy implementation is provided and used by default so the project runs with
no extra dependencies.

References
----------
* T. Lyons, "Rough paths, signatures and the modelling of functions on
  streams", ICM 2014.
* P. Kidger, T. Lyons, "Signatory: differentiable signature computations".
* I. Perez Arribas, "Signatures in machine learning and finance" and the
  "signature payoffs" literature (Lyons, Nejad, Perez Arribas 2019).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # optional acceleration; the project must run without it
    import iisignature  # type: ignore
    HAVE_IISIGNATURE = True
except Exception:  # pragma: no cover - environment dependent
    iisignature = None
    HAVE_IISIGNATURE = False


# ---------------------------------------------------------------------------
# Pure-NumPy truncated signature (tensor-algebra implementation)
# ---------------------------------------------------------------------------

def _tensor_exp(increment: np.ndarray, level: int) -> list[np.ndarray]:
    """Signature of a single straight segment = truncated tensor exponential
    ``(1, x, x⊗x/2!, x⊗x⊗x/3!, ...)`` of the increment ``x``.
    """
    terms: list[np.ndarray] = [np.array(1.0)]
    power = np.array(1.0)                     # x^{⊗0}
    for k in range(1, level + 1):
        power = np.multiply.outer(power, increment)   # x^{⊗k}
        terms.append(power / math.factorial(k))
    return terms


def _chen(a: list[np.ndarray], b: list[np.ndarray],
          level: int) -> list[np.ndarray]:
    """Chen product (truncated tensor-algebra multiplication).

    ``C_m = sum_{i + j = m} A_i ⊗ B_j``; the order (A first, B second) encodes
    the temporal order of the concatenated path segments.
    """
    out: list[np.ndarray] = []
    for m in range(level + 1):
        acc = None
        for i in range(m + 1):
            j = m - i
            term = np.multiply.outer(a[i], b[j])
            acc = term if acc is None else acc + term
        out.append(acc)
    return out


def signature(path: np.ndarray, level: int = 3) -> list[np.ndarray]:
    """Truncated signature of ``path`` (shape ``(n_points, d)``) to ``level``.

    Returns a list ``[S_0, S_1, ..., S_level]`` where ``S_k`` is a rank-``k``
    tensor of shape ``(d,) * k`` (``S_0`` is the scalar ``1``).
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must have shape (n_points, d)")
    increments = np.diff(path, axis=0)

    # Identity element of the truncated tensor algebra.
    d = path.shape[1]
    sig: list[np.ndarray] = [np.array(1.0)] + [
        np.zeros((d,) * k) for k in range(1, level + 1)
    ]
    for inc in increments:
        sig = _chen(sig, _tensor_exp(inc, level), level)
    return sig


def signature_features(path: np.ndarray, level: int = 3,
                       include_zeroth: bool = False) -> np.ndarray:
    """Flatten the truncated signature into a 1-D feature vector.

    Level 0 (the constant ``1``) is dropped by default so a regression can own
    its intercept. Ordering is lexicographic in the tensor indices, consistent
    across calls (this is what matters for the regression, not matching any
    external convention).
    """
    sig = signature(path, level)
    parts = sig if include_zeroth else sig[1:]
    return np.concatenate([np.ravel(p) for p in parts])


def signature_dim(d: int, level: int, include_zeroth: bool = False) -> int:
    """Length of the truncated-signature feature vector for dimension ``d``."""
    total = sum(d ** k for k in range(0, level + 1))
    return total if include_zeroth else total - 1


# ---------------------------------------------------------------------------
# Signature-based (model-free style) pricing of an arithmetic Asian option
# ---------------------------------------------------------------------------

@dataclass
class SignaturePricingResult:
    level: int
    d: int
    n_features: int
    r2_train: float
    r2_test: float
    mc_price: float
    mc_se: float
    sig_price: float
    strike: float
    scatter_realized: np.ndarray   # test realised payoffs (subsample)
    scatter_predicted: np.ndarray  # test predicted payoffs (subsample)
    expected_signature: np.ndarray


def _simulate_gbm_paths(S0: float, r: float, sigma: float, T: float,
                        n_steps: int, n_paths: int, rng) -> np.ndarray:
    """Risk-neutral GBM price paths, shape ``(n_paths, n_steps + 1)``."""
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    incr = (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.log(S0) + np.cumsum(incr, axis=1)
    S = np.hstack([np.full((n_paths, 1), S0), np.exp(log_paths)])
    return S


def _time_augmented_features(S: np.ndarray, T: float, S0: float,
                             level: int) -> np.ndarray:
    """Build the signature feature matrix for a batch of price paths.

    Each path is time-augmented and normalised to ``(t / T, S / S0)`` so the
    two channels are O(1); time augmentation is what lets low-level signature
    terms capture running integrals such as the Asian average.
    """
    n_paths, n_pts = S.shape
    t = np.linspace(0.0, 1.0, n_pts)
    feats = []
    for p in range(n_paths):
        aug = np.column_stack([t, S[p] / S0])
        feats.append(signature_features(aug, level=level))
    return np.asarray(feats)


def price_asian_with_signatures(
    S0: float = 100.0, K: float = 100.0, r: float = 0.03, sigma: float = 0.2,
    T: float = 1.0, n_steps: int = 50, n_paths: int = 6000, level: int = 3,
    seed: int | None = 0,
) -> SignaturePricingResult:
    """Learn an arithmetic-Asian payoff as a linear functional of the path
    signature, then price it as a linear functional of the expected signature.

    Steps
    -----
    1. Simulate risk-neutral GBM paths and split into train / test halves.
    2. Features = truncated signature of the time-augmented, normalised path.
    3. Fit ``discounted_payoff ~ intercept + <l, signature>`` by least squares
       on the training half (this *learns* the payoff object ``l``).
    4. Report in / out-of-sample R^2 to show the linear-in-signature fit.
    5. Price two ways on the test half and compare:
         * plain Monte-Carlo mean of discounted payoffs;
         * signature price = ``intercept + <l, E[S(X)]>`` using the *expected*
           test signature -- the model-free linear functional.
    """
    rng = np.random.default_rng(seed)
    S = _simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, rng)

    disc = math.exp(-r * T)
    average = S.mean(axis=1)                      # arithmetic running average
    payoff = disc * np.maximum(average - K, 0.0)

    X = _time_augmented_features(S, T, S0, level)
    n_train = n_paths // 2
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = payoff[:n_train], payoff[n_train:]

    # Design matrices with an explicit intercept column.
    Atr = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    Ate = np.hstack([np.ones((len(Xte), 1)), Xte])
    coef, *_ = np.linalg.lstsq(Atr, ytr, rcond=None)

    def r2(A, y):
        pred = A @ coef
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    r2_train = r2(Atr, ytr)
    r2_test = r2(Ate, yte)

    # Plain Monte-Carlo price and standard error (test half).
    mc_price = float(yte.mean())
    mc_se = float(yte.std(ddof=1) / math.sqrt(len(yte)))

    # Signature price = intercept + <l, E[S(X)]> with the expected signature
    # estimated on the test half. Equivalent to coef . mean(design row).
    expected_row = Ate.mean(axis=0)
    sig_price = float(expected_row @ coef)
    expected_signature = Xte.mean(axis=0)

    pred_te = Ate @ coef
    # Subsample for a readable scatter plot.
    idx = rng.choice(len(yte), size=min(400, len(yte)), replace=False)

    return SignaturePricingResult(
        level=level, d=2, n_features=X.shape[1],
        r2_train=r2_train, r2_test=r2_test,
        mc_price=mc_price, mc_se=mc_se, sig_price=sig_price, strike=K,
        scatter_realized=yte[idx], scatter_predicted=pred_te[idx],
        expected_signature=expected_signature,
    )


def _smoke_test() -> None:
    """Headless self-check: dimension formula, fallback path, and pricing."""
    # Dimension check against the closed-form (d^{M+1}-1)/(d-1).
    for d, M in [(2, 3), (3, 2), (2, 2)]:
        path = np.cumsum(np.random.default_rng(0).standard_normal((11, d)), axis=0)
        feat = signature_features(path, level=M)
        assert feat.size == signature_dim(d, M), (feat.size, signature_dim(d, M))

    # Chen consistency: signature of a single straight segment equals its
    # tensor exponential.
    seg = np.array([[0.0, 0.0], [0.3, -0.2]])
    sig = signature(seg, level=3)
    texp = _tensor_exp(seg[1] - seg[0], 3)
    for a, b in zip(sig, texp):
        assert np.allclose(a, b)

    res = price_asian_with_signatures(n_paths=2000, n_steps=30, level=3, seed=1)
    assert np.isfinite(res.sig_price) and np.isfinite(res.mc_price)
    assert res.r2_test > 0.5, f"signature fit too weak: R2={res.r2_test:.3f}"
    print(f"[signatures] OK  iisignature={HAVE_IISIGNATURE}  "
          f"feat_dim={res.n_features}  R2_test={res.r2_test:.3f}  "
          f"MC={res.mc_price:.3f}  Sig={res.sig_price:.3f}")


if __name__ == "__main__":
    _smoke_test()
