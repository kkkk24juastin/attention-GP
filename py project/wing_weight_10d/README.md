# Wing Weight 10D Nonstationary Experiment

This directory implements the 4.5 supplementary engineering benchmark case.
It follows the same structure as the existing 2D, 5D, and battery experiments:

- `config.py`: experiment constants, bounds, method registry, and scheduler scope.
- `core.py`: Wing Weight function, nonstationary penalties, TGP prediction, metrics, and summaries.
- `generate_data.py`: shared LHS data generation.
- `run_experiments.py`: PM, LHS, G-opt, D-opt, K-means, EI, UCB, and IMSE sampling scheduler.
- `run_ga_optimization.py`: reads saved training sets, runs archive-style direct-btgp GA, and reports grouped PM-framework QL.
- `summarize_results.py`: rebuilds `refactored_results/summary_statistics.xlsx` from GA results.
- `methods/`: one sampling method per module.

The base 10D Wing Weight formula follows the standard aerospace surrogate
modeling benchmark described by UQTestFuns, OpenTURNS, and the Virtual Library
of Simulation Experiments. Input variables are:

- `Sw`: wing area.
- `Wfw`: fuel weight in the wing.
- `A`: aspect ratio.
- `Lambda`: quarter-chord sweep angle.
- `q`: dynamic pressure.
- `lambda`: taper ratio. The column keeps the literature symbol; Python code
  uses `taper` internally.
- `tc`: thickness-to-chord ratio.
- `Nz`: ultimate load factor.
- `Wdg`: flight design gross weight.
- `Wp`: paint weight.

To address the high-dimensional nonstationary review concern, `core.py` adds
piecewise engineering penalties on top of the base formula:

- structural reinforcement when both `Nz` and `q` are high;
- a sweep-angle regime correction for a narrow `Lambda` interval;
- a thin-wing/high-aspect buckling penalty;
- a small multi-modal aerodynamic interaction term.

The experiment is intentionally split into two stages:

1. `run_experiments.py` only performs active learning and saves each final
   training set, including the initial points and the active-learning points.
   Files are written to `refactored_results/selected_datasets/`, and the index
   is written to `refactored_results/selected_datasets_index.xlsx`.
2. `run_ga_optimization.py` reads those saved training sets, runs archive-style
   direct-btgp GA for each method/repeat, and then re-evaluates all GA points
   together with the PM training set for that repeat. The reported main QL
   follows this grouped PM framework with `PM_EVAL_SEED = 42`.
   The formal workbook is written to `refactored_results/ga_optimization_results.xlsx`.

GA results include:

- `RMSE_all`;
- `RMSE_target`;
- test-set `best_pred_quality_loss` as a discrete QL reference;
- test-set `best_true_target_error`;
- test-set `best_true_response`;
- the selected `best_x1` to `best_x10` coordinates;
- GA optimized `ga_pred_quality_loss` as the main optimization QL under the PM framework;
- GA optimized `ga_self_pred_quality_loss` as the same GA point's direct-btgp self-evaluation;
- PM recheck columns `pm_eval_QL`, `pm_eval_mean`, `pm_eval_variance`;
- `ga_repeat`;
- GA optimized `ga_true_quality_loss`;
- GA optimized `ga_true_target_error`;
- GA optimized `ga_true_response`;
- GA optimized `ga_x1` to `ga_x10` coordinates.

The GA optimization step is intentionally analogous to the rocket case: each
saved training set is used to fit a direct btgp model, and GA searches the
design space for the point minimizing `(mu - T)^2 + variance`. The resulting
designs are then re-evaluated together with the PM training set, and that
PM-framework QL becomes the formal reported `ga_pred_quality_loss`. The true
analytical Wing Weight response is still reported separately.

Run from this directory:

```bash
python run_experiments.py
```

Run GA optimization after training sets have been saved:

```bash
python run_ga_optimization.py
```

Rebuild summaries only:

```bash
python summarize_results.py
```
