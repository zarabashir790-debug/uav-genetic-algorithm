"""Experiment framework and statistics (Step 5)."""

import csv
import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from uav_ga import experiments as ex
from uav_ga.run_experiments import make_figures


def test_mean_std_uses_sample_sd():
    assert ex.mean_std([1.0, 2.0, 3.0, 4.0]) == pytest.approx((2.5, np.std([1, 2, 3, 4], ddof=1)))
    assert ex.mean_std([5.0]) == (5.0, 0.0)
    assert ex.mean_std([1.0, ""]) == (1.0, 0.0)  # blanks ignored


def test_sign_flip_p_value_exact_cases():
    assert ex.sign_flip_p([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)  # all same sign: 2 of 32 flips
    assert ex.sign_flip_p([1, -1]) == pytest.approx(1.0)
    assert ex.sign_flip_p([0, 0, 0]) == pytest.approx(1.0)
    assert 0 < ex.sign_flip_p([0.3, 0.5, -0.1, 0.4, 0.2, 0.6]) < 0.2


def test_paired_difference_aligns_by_seed():
    a = [{"seed": 2, "best_fitness": 0.9}, {"seed": 1, "best_fitness": 0.8}]
    b = [{"seed": 1, "best_fitness": 0.7}, {"seed": 2, "best_fitness": 0.95}]
    p = ex.paired(a, b)
    assert p["n"] == 2 and p["wins_a"] == 1
    assert p["diffs"] == pytest.approx([0.1, -0.05]) and p["mean_diff"] == pytest.approx(0.025)


def test_job_list_covers_all_configs_and_seeds():
    jobs = ex.build_jobs((1, 2), 100)
    ga = [j for j in jobs if j["algorithm"] == "GA"]
    configs = {tuple(sorted(j["params"].items())) for j in ga}
    assert len(configs) == 7 and len(ga) == 14  # default + 2 alternatives for each of 3 factors
    assert sum(j["algorithm"] == "RS" for j in jobs) == 2
    assert all(j["budget"] == 100 and j["env_seed"] == 42 for j in jobs)
    for factor, levels in ex.LEVELS.items():
        for lv in levels:
            assert tuple(sorted({**ex.DEFAULTS, factor: lv}.items())) in configs


def test_checkpoints():
    assert ex.checkpoints(100) == list(range(10, 101, 10))
    assert ex.checkpoints(105)[-1] == 105


@pytest.fixture(scope="module")
def tiny():
    seeds, budget = (1, 2), 70
    rows, curves = ex.run_all(seeds, budget, ex.ENV_SEED, workers=1)
    return seeds, budget, rows, curves


def test_every_run_uses_exactly_the_budget(tiny):
    seeds, budget, rows, curves = tiny
    assert len(rows) == 7 * 2 + 2
    assert all(r["evaluations"] == budget for r in rows)
    assert all(r["feasible"] in (0, 1) for r in rows)
    assert all(len(c) == len(ex.checkpoints(budget)) for c in curves.values())


def test_runs_are_deterministic_and_parallel_matches_serial(tiny):
    seeds, budget, rows, _ = tiny
    again, _ = ex.run_all(seeds, budget, ex.ENV_SEED, workers=2)
    key = lambda r: (r["algorithm"], str(r["population_size"]), str(r["mutation_prob"]),
                     str(r["tournament_size"]), r["seed"])
    a, b = sorted(rows, key=key), sorted(again, key=key)
    assert [(r["best_fitness"], r["coverage"], r["generations"]) for r in a] == \
           [(r["best_fitness"], r["coverage"], r["generations"]) for r in b]


def test_curves_are_nondecreasing_best_so_far(tiny):
    for key, c in tiny[3].items():
        vals = [v for v in c if v is not None]
        assert all(y >= x - 1e-12 for x, y in zip(vals, vals[1:])), key
    rs = [c for k, c in tiny[3].items() if k.startswith("RS")]
    assert all(v is not None for c in rs for v in c)


def test_curve_end_value_equals_run_best_fitness(tiny):
    _, _, rows, curves = tiny
    for r in rows:
        assert curves[ex.curve_key(r)][-1] == pytest.approx(r["best_fitness"])


def test_summaries_match_manual_statistics(tiny):
    _, _, rows, _ = tiny
    default = ex.select(rows, "GA", **ex.DEFAULTS)
    s = ex.summarize(default)
    fit = [r["best_fitness"] for r in default]
    assert s["best_fitness_mean"] == pytest.approx(np.mean(fit))
    assert s["best_fitness_std"] == pytest.approx(np.std(fit, ddof=1))
    assert s["best_fitness_max"] == max(fit) and s["n"] == 2
    table = ex.sensitivity_tables(rows)
    assert len(table) == 9 and sum(e["is_default"] for e in table) == 3
    pop20 = next(e for e in table if e["factor"] == "population_size" and e["level"] == 20)
    pr = ex.paired(ex.select(rows, "GA", **{**ex.DEFAULTS, "population_size": 20}), default)
    assert pop20["vs_default_mean_diff"] == pytest.approx(pr["mean_diff"])


def test_results_and_figures_are_saved_and_consistent(tiny, tmp_path):
    seeds, budget, rows, curves = tiny
    meta = {"env_seed": 42, "seeds": list(seeds), "budget": budget, "defaults": ex.DEFAULTS}
    files = ex.save_results(tmp_path, rows, curves, meta)
    figs = make_figures(rows, curves, meta, tmp_path)
    assert len(figs) == 6
    for f in list(figs.values()):
        data = open(f, "rb").read()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5000
    saved = list(csv.DictReader(open(files["raw_runs"])))
    assert len(saved) == len(rows)
    assert [float(r["best_fitness"]) for r in saved] == pytest.approx([r["best_fitness"] for r in rows])
    comp = list(csv.DictReader(open(files["comparison_summary"])))
    assert comp[-1]["algorithm"] == "GA minus RS (paired)"
    stored = json.load(open(files["progress_curves"]))
    assert stored["checkpoints"] == ex.checkpoints(budget) and len(stored["curves"]) == len(rows)
