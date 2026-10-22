"""Tests for conformal.py on SYNTHETIC data only (RQ3 step 2). Run: python test_conformal.py"""
import numpy as np
import conformal as cf

ALPHA = 0.10


def test_quantile_index():
    """k = ceil((n+1)(1-alpha)): n = 8 is too small at 90% (set = all values); n = 9 -> 9th smallest
    (0.9 * 10 is 9.000000000000002 in floating point, which a naive ceil would turn into 10)."""
    assert cf.split_threshold(np.arange(8.0), ALPHA) == np.inf
    assert cf.split_threshold(np.arange(9.0), ALPHA) == 8.0
    assert cf.split_threshold(np.arange(19.0), ALPHA) == 17.0          # k = ceil(20 * 0.9) = 18
    assert cf.k_lower(9, ALPHA) == 1 and cf.k_upper(9, ALPHA) == 9
    print("ok  quantile index ceil((n+1)(1-alpha)), safe against float error; too few points -> all values")


def test_pvalue_set_equals_quantile_set():
    """y in set  <=>  p(y) > alpha  <=>  s(y) <= split threshold (incl. ties)."""
    rng = np.random.default_rng(0)
    for n in [5, 9, 10, 37, 100]:
        cal = rng.integers(0, 5, n).astype(float)                       # many ties
        test = rng.integers(0, 6, 500).astype(float)
        t = cf.split_threshold(cal, ALPHA)
        assert np.array_equal(cf.p_values(cal, test) > ALPHA, test <= t), n
    print("ok  p-value rule and quantile rule give identical sets (with ties, several n)")


def test_split_conformal_coverage():
    """i.i.d. continuous scores: coverage of a new point is between 1-alpha and 1-alpha+1/(n+1)."""
    rng = np.random.default_rng(1)
    R, n = 4000, 40
    cal = np.abs(rng.normal(size=(R, n)))
    new = np.abs(rng.normal(size=R))
    t = np.sort(cal, axis=1)[:, cf.k_upper(n, ALPHA) - 1]
    cov = (new <= t).mean()
    assert 0.89 <= cov <= 0.92, cov
    print(f"ok  split conformal on i.i.d. data covers {cov:.3f} (theory 0.900-0.924)")


def test_one_per_student():
    rng = np.random.default_rng(2)
    students = np.repeat(np.arange(50), rng.integers(1, 30, 50))
    idx = cf.one_per_student(students, rng)
    assert len(idx) == 50 and sorted(students[idx]) == list(range(50))
    counts = np.zeros(4)                                                 # uniform within a student
    s4 = np.array([7, 7, 7, 7, 3])
    for _ in range(20000):
        counts[[i for i in cf.one_per_student(s4, rng) if s4[i] == 7][0]] += 1
    assert np.all(np.abs(counts / 20000 - 0.25) < 0.015), counts / 20000
    print("ok  one_per_student: exactly one row per student, uniform within a student")


def test_subsample_threshold_matches_definition():
    """The threshold shortcut gives the same sets as averaging the B p-values directly."""
    rng_data = np.random.default_rng(3)
    students = np.repeat(np.arange(30), rng_data.integers(1, 20, 30))
    scores = rng_data.integers(0, 6, len(students)).astype(float) + rng_data.random(len(students)) * 0.5
    test = np.linspace(-0.5, 7, 200)
    for B in [1, 7, 50]:
        t = cf.subsample_threshold(scores, students, ALPHA, B, np.random.default_rng(10))
        rng = np.random.default_rng(10)
        mean_p = np.mean([cf.p_values(scores[cf.one_per_student(students, rng)], test) for _ in range(B)], 0)
        assert np.array_equal(mean_p > ALPHA, test <= t), B
    print("ok  repeated-subsampling threshold equals the averaged-p-value definition (B = 1, 7, 50)")


def hierarchical(rng, k, heavy_n=100, light_n=5):
    """Synthetic students: half 'heavy' (many cutoffs, small errors), half 'light' (few cutoffs,
    large errors). Survey-level calibration is dominated by the heavy students."""
    heavy = rng.random(k) < 0.5
    n = np.where(heavy, heavy_n, light_n)
    sd = np.where(heavy, 0.3, 2.0)
    students = np.repeat(np.arange(k), n)
    scores = np.abs(rng.normal(size=len(students)) * np.repeat(sd, n))
    return students, scores


def test_student_level_vs_survey_level():
    """New student, random cutoff: survey-level calibration (A) under-covers badly; repeated
    subsampling (B) keeps >= 1 - 2 alpha and in practice about 1 - alpha; pooled CDF is close."""
    rng = np.random.default_rng(4)
    covA, covB, covB1, covP = [], [], [], []
    for _ in range(300):
        students, scores = hierarchical(rng, 60)
        new_sd = 0.3 if rng.random() < 0.5 else 2.0                      # a new random student
        new = abs(rng.normal() * new_sd)
        covA.append(new <= cf.split_threshold(scores, ALPHA))
        covB.append(new <= cf.subsample_threshold(scores, students, ALPHA, 50, rng))
        covB1.append(new <= cf.subsample_threshold(scores, students, ALPHA, 1, rng))
        covP.append(new <= cf.pooled_cdf_threshold(scores, students, ALPHA))
    a, b, b1, p = map(np.mean, (covA, covB, covB1, covP))
    assert a < 0.75 and b >= 0.87 and b1 >= 0.86 and 0.84 <= p <= 0.97, (a, b, b1, p)
    print(f"ok  new-student coverage: survey-level A {a:.2f} (fails), repeated subsampling B {b:.2f}, "
          f"single subsample {b1:.2f}, pooled CDF {p:.2f}")


def test_cvplus_brute_force():
    """cvplus_bounds equals a direct reading of Barber et al. eq. (11) on a tiny example."""
    rng = np.random.default_rng(5)
    n, K = 7, 3
    fold = np.array([0, 0, 1, 1, 1, 2, 2])
    resid = rng.random(n) * 2
    preds = rng.random((4, K)) * 6
    for alpha in [0.2, 0.3]:
        lo, hi = cf.cvplus_bounds(preds, fold, resid, alpha)
        for x in range(4):
            v_lo = sorted(preds[x, fold[i]] - resid[i] for i in range(n))
            v_hi = sorted(preds[x, fold[i]] + resid[i] for i in range(n))
            kl, ku = int(np.floor(alpha * (n + 1) + 1e-9)), int(np.ceil((1 - alpha) * (n + 1) - 1e-9))
            assert np.isclose(lo[x], v_lo[kl - 1] if kl >= 1 else -np.inf)
            assert np.isclose(hi[x], v_hi[ku - 1] if ku <= n else np.inf)
    print("ok  CV+ bounds match a brute-force reading of Barber et al. eq. (11)")


def test_cvplus_with_identical_models_is_split():
    """If every fold model predicts the same, CV+ reduces to split conformal: y_hat +- threshold."""
    rng = np.random.default_rng(6)
    resid = np.abs(rng.normal(size=99))
    fold = rng.integers(0, 4, 99)
    y_hat = rng.random(10) * 6
    lo, hi = cf.cvplus_bounds(np.repeat(y_hat[:, None], 4, 1), fold, resid, ALPHA)
    t = cf.split_threshold(resid, ALPHA)
    assert np.allclose(lo, y_hat - t) and np.allclose(hi, y_hat + t)
    print("ok  CV+ with identical fold models = split conformal")


def test_cvplus_coverage():
    """CV+ around K-fold linear fits on i.i.d. data: at least 1 - 2 alpha, in practice about 1 - alpha."""
    rng = np.random.default_rng(7)
    cov = []
    for _ in range(400):
        x = rng.random(60) * 4
        y = 1 + 0.8 * x + rng.normal(size=60)
        fold = np.arange(60) % 4
        coefs = [np.polyfit(x[fold != k], y[fold != k], 1) for k in range(4)]
        resid = np.abs(y - np.array([np.polyval(coefs[f], xi) for f, xi in zip(fold, x)]))
        x_new = rng.random() * 4
        y_new = 1 + 0.8 * x_new + rng.normal()
        lo, hi = cf.cvplus_bounds(np.array([[np.polyval(c, x_new) for c in coefs]]), fold, resid, ALPHA)
        cov.append(lo[0] <= y_new <= hi[0])
    c = np.mean(cov)
    assert c >= 0.86, c
    print(f"ok  CV+ coverage on synthetic linear data {c:.2f} (theory >= {1 - 2 * ALPHA - 2 * 0.75 / 16:.2f})")


def test_mondrian():
    """Two groups with different error sizes: pooled calibration under-covers the noisier minority
    group; per-group calibration covers both. A group's threshold ignores other groups' rows."""
    rng = np.random.default_rng(8)
    pooled_cov, group_cov = {0: [], 1: []}, {0: [], 1: []}
    for _ in range(300):
        k = 120
        g_stu = (rng.random(k) < 0.25).astype(int)                       # group 1 = 25%, noisier
        n = rng.integers(3, 30, k)
        students = np.repeat(np.arange(k), n)
        groups = np.repeat(g_stu, n)
        scores = np.abs(rng.normal(size=len(students)) * np.where(groups == 1, 3.0, 1.0))
        t_pool = cf.subsample_threshold(scores, students, ALPHA, 20, rng)
        t_grp = cf.group_thresholds(cf.subsample_threshold, scores, groups, [0, 1],
                                    students=students, alpha=ALPHA, B=20, rng=rng)
        for g, sd in [(0, 1.0), (1, 3.0)]:
            new = abs(rng.normal() * sd)
            pooled_cov[g].append(new <= t_pool)
            group_cov[g].append(new <= t_grp[g])
    p1, m0, m1 = np.mean(pooled_cov[1]), np.mean(group_cov[0]), np.mean(group_cov[1])
    assert p1 < 0.80 and m0 >= 0.87 and m1 >= 0.87, (p1, m0, m1)
    # invariance: changing group 1's scores does not change group 0's threshold
    s2 = scores.copy(); s2[groups == 1] *= 10
    a = cf.group_thresholds(cf.split_threshold, scores, groups, [0], alpha=ALPHA)[0]
    b = cf.group_thresholds(cf.split_threshold, s2, groups, [0], alpha=ALPHA)[0]
    assert a == b
    print(f"ok  per-group calibration: pooled covers the noisy group {p1:.2f}; per-group covers "
          f"{m0:.2f} / {m1:.2f}; a group's threshold uses only its own rows")


def test_joint_coverage():
    """4 horizons: alpha / 4 per horizon (Bonferroni) gives joint coverage >= 1 - alpha; plain alpha
    per horizon does not. The max-normalised score (extra) reaches the target with smaller sets."""
    rng = np.random.default_rng(9)
    R, n, H = 1500, 80, 4
    sd = np.array([0.5, 1.0, 1.5, 2.0])
    cal = np.abs(rng.normal(size=(R, n, H)) * sd)
    new = np.abs(rng.normal(size=(R, H)) * sd)
    t_bon = np.sort(cal, axis=1)[:, cf.k_upper(n, ALPHA / H) - 1, :]
    t_plain = np.sort(cal, axis=1)[:, cf.k_upper(n, ALPHA) - 1, :]
    j_bon, j_plain = np.all(new <= t_bon, 1).mean(), np.all(new <= t_plain, 1).mean()
    mask = np.ones((n, H), bool)
    j_max, w_bon, w_max = [], t_bon.sum(1).mean(), []
    for r in range(R):
        s = cf.max_norm_score(cal[r], np.zeros((n, H)), mask, sd)
        t = cf.split_threshold(s, ALPHA)
        j_max.append(np.all(new[r] <= t * sd))
        w_max.append((t * sd).sum())
    j_max, w_max = np.mean(j_max), np.mean(w_max)
    assert j_bon >= 0.89 and j_plain < 0.80 and 0.88 <= j_max <= 0.93 and w_max < w_bon, \
        (j_bon, j_plain, j_max, w_max, w_bon)
    cov = np.array([[1, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]], bool)
    msk = np.array([[1, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]], bool)
    jc = cf.joint_covered(cov, msk)
    assert jc[0] == 1 and jc[1] == 1 and np.isnan(jc[2])
    print(f"ok  joint coverage: Bonferroni {j_bon:.2f}, no correction {j_plain:.2f} (fails), "
          f"max-normalised extra {j_max:.2f} with {100 * (1 - w_max / w_bon):.0f}% narrower sets; "
          "missing horizons are ignored")


def test_cqr():
    """CQR score is negative inside [q_lo, q_hi]; the set {s(y) <= t} equals the integers in
    [q_lo - t, q_hi + t]; a negative t can shrink the set (even to empty)."""
    rng = np.random.default_rng(11)
    q_lo, q_hi = rng.random(300) * 3, 3 + rng.random(300) * 3
    assert np.all(cf.cqr_score((q_lo + q_hi) / 2, q_lo, q_hi) < 0)
    for t in [-1.0, 0.0, 0.7, np.inf]:
        sets = cf.sets_from_threshold(cf.cqr_candidates(q_lo, q_hi), t)
        lo, hi = q_lo - t, q_hi + t
        direct = (cf.Y_VALUES[None, :] >= lo[:, None] - 1e-9) & (cf.Y_VALUES[None, :] <= hi[:, None] + 1e-9)
        assert np.array_equal(sets, direct), t
    print("ok  CQR: score sign, set = integers in [q_lo - t, q_hi + t]")


def test_sets_are_intervals_and_clipping_is_harmless():
    """Sets of 0..6 from absolute or CQR scores have no holes; clipping to [0, 6] never changes
    whether an integer score 0..6 is covered; set_summary reports lowest, highest, size."""
    rng = np.random.default_rng(12)
    y_hat = rng.random(500) * 8 - 1
    for t in [0.3, 1.0, 2.5]:
        sets = cf.sets_from_threshold(cf.abs_candidates(y_hat), t)
        lowest, highest, size = cf.set_summary(sets)
        ok = size > 0
        assert np.all(highest[ok] - lowest[ok] + 1 == size[ok])          # contiguous
        lo, hi, width = cf.abs_interval(y_hat, t)
        y = rng.integers(0, 7, 500).astype(float)
        unclipped = (y >= y_hat - t - 1e-9) & (y <= y_hat + t + 1e-9)
        assert np.array_equal(cf.covered(sets, y), unclipped)
        assert np.all((lo >= 0) & (hi <= 6) & (width >= 0))
    assert np.isinf(cf.abs_interval(np.array([3.0]), np.inf)[0]).sum() == 0     # clipped to [0, 6]
    print("ok  sets have no holes; clipping to [0, 6] never changes coverage; summaries consistent")


def test_integer_outcomes_over_cover():
    """With integer scores (e.g. persistence residuals) ties push coverage ABOVE the target; this is
    reported, not randomised away."""
    rng = np.random.default_rng(13)
    R, n = 3000, 60
    cal = rng.choice([0, 1, 2, 3], size=(R, n), p=[0.5, 0.3, 0.15, 0.05]).astype(float)
    new = rng.choice([0, 1, 2, 3], size=R, p=[0.5, 0.3, 0.15, 0.05]).astype(float)
    t = np.sort(cal, axis=1)[:, cf.k_upper(n, ALPHA) - 1]
    cov = (new <= t).mean()
    assert cov >= 0.92, cov
    print(f"ok  integer scores: coverage {cov:.2f} at a 90% target (over-coverage from ties, as expected)")


def test_student_mean():
    v = np.array([1, 1, 1, 0, np.nan, 0.0])
    s = np.array(["a", "a", "a", "b", "b", "c"])
    assert np.isclose(cf.student_mean(v, s), (1 + 0 + 0) / 3)
    print("ok  student-weighted mean (each student counts once; nan ignored)")


if __name__ == "__main__":
    test_quantile_index()
    test_pvalue_set_equals_quantile_set()
    test_split_conformal_coverage()
    test_one_per_student()
    test_subsample_threshold_matches_definition()
    test_student_level_vs_survey_level()
    test_cvplus_brute_force()
    test_cvplus_with_identical_models_is_split()
    test_cvplus_coverage()
    test_mondrian()
    test_joint_coverage()
    test_cqr()
    test_sets_are_intervals_and_clipping_is_harmless()
    test_integer_outcomes_over_cover()
    test_student_mean()
    print("all tests passed")
