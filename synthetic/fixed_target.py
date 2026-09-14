"""Conditional decomposition for repeated estimates of ONE fixed target.

Never pass a stack of targets. Different targets require separate calls. An
explicit average of those conditional risks may be reported as a task average,
but pooled estimator variance is not the within-target estimation variance.
"""
import hashlib
import numpy as np


def target_id(theta_star):
    target = np.asarray(theta_star, dtype=np.float64)
    if target.ndim != 1 or not np.all(np.isfinite(target)):
        raise ValueError('theta_star must be one finite, fixed vector, not per-trial targets')
    return hashlib.sha256(target.astype('<f8').tobytes()).hexdigest()


def fixed_target_decomposition(estimates, theta_star, risks=None):
    """Return empirical risk, squared-bias term, variance (all with factor 1/2).

    Variance uses M, so bias + variance equals the empirical mean risk exactly.
    This finite-M plug-in bias is not an unbiased estimate of population bias.
    Optional per-trial risks must agree with distances to this same target.
    """
    target = np.asarray(theta_star, dtype=np.float64)
    target_id(target)  # validate before any broadcasting
    stacked = np.asarray(estimates, dtype=np.float64)
    if stacked.ndim != 2 or stacked.shape[1:] != target.shape or not len(stacked):
        raise ValueError('estimates must have shape (M, D) for the one D-vector target')
    if not np.all(np.isfinite(stacked)):
        raise ValueError('estimates contain nonfinite values')
    risk_each = .5*np.sum((stacked-target)**2, axis=1)
    if risks is not None:
        saved = np.asarray(risks, dtype=np.float64)
        if saved.shape != risk_each.shape or not np.allclose(saved, risk_each, rtol=1e-9, atol=1e-12):
            raise ValueError('Per-trial risks do not use the same fixed target/metric; do not pool targets')
    mean = stacked.mean(axis=0)
    bias = .5*float(np.sum((mean-target)**2))
    variance = .5*float(np.mean(np.sum((stacked-mean)**2, axis=1)))
    risk = float(risk_each.mean())
    if not np.isclose(risk, bias+variance, rtol=1e-10, atol=1e-12):
        raise ValueError('Fixed-target risk decomposition failed')
    return risk, bias, variance
