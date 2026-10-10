import numpy as np, pytest
from backend.ml.metrics import compute_metrics, choose_threshold
from backend.ml.fairness import group_table, disparities, wilson


def test_metrics_known_values():
    y = np.array([1, 1, 0, 0, 1, 0]); p = np.array([.9, .4, .2, .6, .8, .1])
    m = compute_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == [[2, 1], [1, 2]] and m["recall"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(2 / 3)


def test_threshold_keeps_target_recall():
    rng = np.random.RandomState(0); y = rng.randint(0, 2, 500); p = np.clip(y * .3 + rng.rand(500) * .7, 0, 1)
    t, why = choose_threshold(y, p); assert ((p >= t) & (y == 1)).sum() / (y == 1).sum() >= 0.9 and "recall" in why


def test_wilson_bounds():
    lo, hi = wilson(8, 10); assert 0 <= lo < 0.8 < hi <= 1 and wilson(0, 0) == [None, None]


def test_fairness_group_metrics_exact():
    # group A: 40 pos (30 caught) / 60 neg (6 FP); group B: 40 pos (20 caught) / 60 neg (12 FP)
    y = np.r_[np.ones(40), np.zeros(60), np.ones(40), np.zeros(60)]
    pred = np.r_[np.ones(30), np.zeros(10), np.ones(6), np.zeros(54), np.ones(20), np.zeros(20), np.ones(12), np.zeros(48)]
    g = np.r_[["A"] * 100, ["B"] * 100]; t = group_table(y, pred, g); d = disparities(t)
    assert t[0]["tpr"] == pytest.approx(.75) and t[1]["tpr"] == pytest.approx(.5) and t[0]["fpr"] == pytest.approx(.1)
    assert d["equal_opportunity_difference"] == pytest.approx(.25) and d["demographic_parity_difference"] == pytest.approx(.36 - .32)
    assert d["equalized_odds_difference"] == pytest.approx(.25)


def test_small_groups_not_reliable_and_no_parity():
    y = np.array([1, 0] * 10); pred = y.copy(); g = np.array(["A", "B"] * 10)
    t = group_table(y, pred, g); assert not any(r["reliable"] for r in t) and disparities(t) is None
