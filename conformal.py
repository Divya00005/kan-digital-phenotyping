"""
RQ3 conformal methods (pre-registered in CLAUDE.md, 2026-10-22). numpy only: no model or data code,
so any point predictor (persistence, linear, KAN, later KAN-ODE) can be wrapped without changes.

Outcomes are integer scores 0..6 (PHQ-2, GAD-2). A prediction set is the set of score values y in
0..6 whose conformity score s(y) is at most a threshold t:
    absolute residual (Lei et al. 2018):   s(y) = |y - y_hat|             -> interval [y_hat - t, y_hat + t]
    CQR (Romano et al. 2019):              s(y) = max(q_lo - y, y - q_hi)  -> interval [q_lo - t, q_hi + t]
Intervals are clipped to [0, 6]; a forecast is covered if its true integer score is in the set.
No randomised tie-breaking: with integer outcomes, coverage can be above the target.

Thresholds from calibration scores:
    split_threshold       survey-level split conformal, Lei et al. Algorithm 2          (method A)
    subsample_threshold   one cutoff per calibration student, B times, averaged p-values  (method B)
                          (Dunn et al., repeated subsampling: >= 1 - 2 alpha for a random
                          cutoff of a new student; B = 1 = single subsample, >= 1 - alpha)
    pooled_cdf_threshold  each student's residual CDF weighted equally (Dunn et al. Method 1;
                          asymptotic only)
CV+ (Barber et al. eq. 11) is not a single threshold: see cvplus_bounds / cvplus_subsample_sets.
Per-group (Mondrian) calibration (Romano et al. 2020) = any threshold computed from one group's
calibration rows only: group_thresholds. Joint coverage over H horizons (Stankeviciute et al.)
= alpha / H per horizon: joint_covered. Max-normalised joint score: an extra, not from the papers.
"""
import numpy as np

Y_VALUES = np.arange(7, dtype=float)          # possible PHQ-2 / GAD-2 scores
LOW, HIGH = 0.0, 6.0
EPS = 1e-9                                    # guards ceil / floor against float error


# ----------------------------------------------------------------------------- scores
def abs_score(y, y_hat):
    return np.abs(np.asarray(y, float) - np.asarray(y_hat, float))


def cqr_score(y, q_lo, q_hi):
    y = np.asarray(y, float)
    return np.maximum(np.asarray(q_lo, float) - y, y - np.asarray(q_hi, float))


def abs_candidates(y_hat):
    """(n,) predictions -> (n, 7) score s(y) of every possible score value y."""
    return np.abs(Y_VALUES[None, :] - np.asarray(y_hat, float)[:, None])


def cqr_candidates(q_lo, q_hi):
    y = Y_VALUES[None, :]
    return np.maximum(np.asarray(q_lo, float)[:, None] - y, y - np.asarray(q_hi, float)[:, None])


# ----------------------------------------------------------------------------- basic conformal
def k_upper(n, alpha):
    """ceil((n + 1)(1 - alpha)), safe against float error (e.g. n = 9, alpha = 0.1 -> 9)."""
    return int(np.ceil((n + 1) * (1 - alpha) - EPS))


def k_lower(n, alpha):
    """floor(alpha (n + 1)), safe against float error."""
    return int(np.floor(alpha * (n + 1) + EPS))


def split_threshold(scores, alpha):
    """Lei et al. Algorithm 2: the k-th smallest calibration score, k = ceil((n+1)(1-alpha));
    +inf (the set is all of 0..6) if k > n."""
    s = np.sort(np.asarray(scores, float))
    k = k_upper(len(s), alpha)
    return np.inf if k > len(s) else float(s[k - 1])


def p_values(cal_scores, test_scores):
    """Conformal p-value of each test score: (1 + #{calibration scores >= s}) / (n + 1)."""
    c = np.sort(np.asarray(cal_scores, float))
    ge = len(c) - np.searchsorted(c, np.asarray(test_scores, float), side="left")
    return (1 + ge) / (len(c) + 1)


def one_per_student(students, rng):
    """Row indices: one row drawn uniformly at random for each distinct student."""
    students = np.asarray(students)
    order = np.lexsort((rng.random(len(students)), students))   # by student, random within
    s = students[order]
    first = np.r_[True, s[1:] != s[:-1]]
    return order[first]


def subsample_threshold(scores, students, alpha, B, rng):
    """Dunn et al. repeated subsampling (split version). In each of B subsamples, one score per
    calibration student; y is in the set if the mean over subsamples of its p-value > alpha.
    The mean p-value only decreases as s grows, so the set is {y : s(y) <= t*}; returns t*.
    B = 1 gives the single-subsample method (exact >= 1 - alpha)."""
    scores = np.asarray(scores, float)
    draws = np.stack([np.sort(scores[one_per_student(students, rng)]) for _ in range(B)])
    m = draws.shape[1]
    if 1.0 / (m + 1) > alpha:                 # even above every score the p-value exceeds alpha
        return np.inf
    cand = np.unique(draws)
    ge = m - np.stack([np.searchsorted(d, cand, side="left") for d in draws])
    mean_p = ((1 + ge) / (m + 1)).mean(0)
    ok = cand[mean_p > alpha]
    return float(ok.max()) if len(ok) else -np.inf


def pooled_cdf_threshold(scores, students, alpha):
    """Dunn et al. Method 1: smallest t with (1/K) sum_j F_j(t) >= 1 - alpha, F_j = student j's
    empirical CDF of scores (every student weighted equally). Asymptotically valid only."""
    scores = np.asarray(scores, float)
    u, inv, cnt = np.unique(students, return_inverse=True, return_counts=True)
    w = 1.0 / (len(u) * cnt[inv])
    o = np.argsort(scores, kind="mergesort")
    s, cw = scores[o], np.cumsum(w[o])
    i = int(np.searchsorted(cw, 1 - alpha - EPS, side="left"))
    return float(s[min(i, len(s) - 1)])


def group_thresholds(threshold_fn, scores, groups, wanted, **kw):
    """Per-group (Mondrian) calibration: threshold_fn on the calibration rows of each group only.
    threshold_fn(scores, **kw) or, if kw has 'students', with those rows' students."""
    out = {}
    for g in wanted:
        m = np.asarray(groups) == g
        args = {k: (np.asarray(v)[m] if k == "students" else v) for k, v in kw.items()}
        out[g] = threshold_fn(np.asarray(scores)[m], **args) if m.any() else np.inf
    return out


# ----------------------------------------------------------------------------- CV+
def cvplus_bounds(test_preds, cal_fold, cal_resid, alpha):
    """Barber et al. eq. (11). test_preds (n_test, K): predictions of the K fold models;
    cal_fold (n,): fold of each calibration point; cal_resid (n,): its out-of-fold |residual|.
    lower = floor(alpha (n+1))-th smallest of mu_fold(i)(x) - R_i (or -inf),
    upper = ceil((1 - alpha)(n+1))-th smallest of mu_fold(i)(x) + R_i (or +inf)."""
    n = len(cal_resid)
    mu = np.asarray(test_preds, float)[:, np.asarray(cal_fold)]
    r = np.asarray(cal_resid, float)[None, :]
    kl, ku = k_lower(n, alpha), k_upper(n, alpha)
    lower = np.sort(mu - r, axis=1)[:, kl - 1] if kl >= 1 else np.full(len(mu), -np.inf)
    upper = np.sort(mu + r, axis=1)[:, ku - 1] if ku <= n else np.full(len(mu), np.inf)
    return lower, upper


def cvplus_subsample_sets(test_preds, cal_fold, cal_resid, cal_students, alpha, B, rng):
    """B + CV+ (empirical comparison only; not covered by either paper's theorem): in each of B
    subsamples, one calibration cutoff per student and the CV+ interval; a score value is in the
    final set if it is inside the CV+ interval in at least half of the subsamples."""
    votes = np.zeros((len(test_preds), len(Y_VALUES)))
    for _ in range(B):
        idx = one_per_student(cal_students, rng)
        lo, hi = cvplus_bounds(test_preds, np.asarray(cal_fold)[idx], np.asarray(cal_resid)[idx], alpha)
        votes += (Y_VALUES[None, :] >= lo[:, None] - EPS) & (Y_VALUES[None, :] <= hi[:, None] + EPS)
    return votes >= B / 2


# ----------------------------------------------------------------------------- sets and intervals
def sets_from_threshold(candidates, t):
    """(n, 7) candidate scores and a threshold (scalar or (n,)) -> (n, 7) boolean set."""
    return np.asarray(candidates) <= np.asarray(t, float)[..., None] + EPS if np.ndim(t) else \
        np.asarray(candidates) <= t + EPS


def clip_interval(lo, hi):
    """Clip to [0, 6]; width 0 if the interval is empty."""
    lo, hi = np.clip(lo, LOW, HIGH), np.clip(hi, LOW, HIGH)
    return lo, hi, np.maximum(hi - lo, 0.0)


def abs_interval(y_hat, t):
    y_hat = np.asarray(y_hat, float)
    return clip_interval(y_hat - t, y_hat + t)


def cqr_interval(q_lo, q_hi, t):
    return clip_interval(np.asarray(q_lo, float) - t, np.asarray(q_hi, float) + t)


def set_summary(sets):
    """(n, 7) boolean -> lowest value, highest value (nan if empty), number of values."""
    sets = np.asarray(sets, bool)
    size = sets.sum(1)
    lowest = np.where(size > 0, sets.argmax(1), np.nan)
    highest = np.where(size > 0, len(Y_VALUES) - 1 - sets[:, ::-1].argmax(1), np.nan)
    return lowest, highest, size


def covered(sets, y_true):
    """Is the true integer score inside the set? (n,) boolean."""
    y = np.rint(np.asarray(y_true, float)).astype(int)
    return np.asarray(sets, bool)[np.arange(len(y)), y]


# ----------------------------------------------------------------------------- joint coverage
def joint_covered(cov, mask):
    """(n, H) per-horizon coverage and target mask -> (n,) 1.0 if every horizon with a target is
    covered, 0.0 if not, nan if the cutoff has no target at all."""
    cov, mask = np.asarray(cov, bool), np.asarray(mask, bool)
    ok = np.all(cov | ~mask, axis=1).astype(float)
    return np.where(mask.any(1), ok, np.nan)


def max_norm_score(y, y_hat, mask, scale):
    """EXTRA (not from the six papers): max over available horizons of |y_h - y_hat_h| / s_h.
    y, y_hat, mask: (n, H); scale: (H,). Rows without any target -> nan."""
    r = np.abs(np.asarray(y, float) - np.asarray(y_hat, float)) / np.asarray(scale, float)[None, :]
    r = np.where(mask, r, -np.inf).max(1)
    return np.where(np.asarray(mask).any(1), r, np.nan)


# ----------------------------------------------------------------------------- summaries
def student_mean(values, students):
    """Mean over students of each student's mean value (nan values ignored): coverage for a random
    cutoff of a random student, the quantity the student-level guarantee is about."""
    values, students = np.asarray(values, float), np.asarray(students)
    ok = ~np.isnan(values)
    u, inv = np.unique(students[ok], return_inverse=True)
    sums = np.bincount(inv, weights=values[ok], minlength=len(u))
    cnt = np.bincount(inv, minlength=len(u))
    return float(np.mean(sums / cnt)) if len(u) else np.nan
