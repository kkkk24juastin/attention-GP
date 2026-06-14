# Battery SOC Prediction Experiment

This directory contains the refactored 4.3 lithium battery SOC prediction case.
The implementation follows the same structure used by the 2D and 5D experiments:

- `config.py`: experiment constants, paths, method registry, and scheduler scope.
- `core.py`: shared data loading, TGP prediction, metric evaluation, result saving, and summaries.
- `generate_data.py`: converts the original Excel files into `generated_data/shared_data.xlsx`.
- `run_experiments.py`: runs PM, LHS, G-opt, D-opt, K-means, EI, UCB, and IMSE.
- `summarize_results.py`: rebuilds `refactored_results/summary_statistics.xlsx`.
- `methods/`: one sampling method per module.
- `archive_legacy/`: archived legacy scripts, plots, and result files.

The original source data files remain in this directory:

- `data.xlsx`: test set.
- `acttrain.xlsx`: fixed initial training set.
- `actpool.xlsx`: candidate pool.
- `lhstrain.xlsx`: legacy LHS final training set.

For consistency with the original battery case, `LHS` directly evaluates
`lhstrain.xlsx`, while the other methods start from `acttrain.xlsx` and add
`N_ADDED = 30` samples from `actpool.xlsx`.

The 4.3 supplementary experiment checklist is implemented by:

- Adding modern active learning / Bayesian optimization baselines:
  `EI`, `UCB`, and `IMSE`.
- Reporting engineering error metrics beyond RMSE:
  `MAE_all`, `MaxAE_all`, `SOC_error_le_5pct_ratio`, and `MaxAE_target`.

Run from this directory:

```bash
python run_experiments.py
```

Rebuild summaries only:

```bash
python summarize_results.py
```
