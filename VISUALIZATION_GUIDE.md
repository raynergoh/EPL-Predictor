# Backtesting Visualization Guide

This document explains the visualization charts generated for backtesting results and how to regenerate them.

## Quick Start

Generate all charts and tables:

```bash
python3 src/backtest/visualize_results.py
```

**Output:**
- 📊 4 PNG charts in `data/backtest/charts/`
- 📋 Markdown table printed to console (for README)

## Generated Charts

### 1. Accuracy by Fold (`accuracy_by_fold.png`)

**Purpose:** Compare baseline vs time-weighted model across all 5 test folds

**Key Features:**
- Side-by-side bar chart
- Value labels on each bar
- Horizontal lines showing averages
- Clear visual comparison

**Insights:**
- Time-weighted wins in 4/5 folds
- Fold 4 (2016-17) is the only outlier
- Consistent ~0.5-1% improvement in most folds

### 2. Metrics Comparison (`metrics_comparison.png`)

**Purpose:** Show all three evaluation metrics side by side

**Metrics:**
- **Accuracy (%):** Percentage of correct predictions
- **Brier Score:** Calibration quality (lower = better)
- **Log-Likelihood:** Probabilistic prediction quality (higher = better)

**Key Features:**
- 3-panel layout for easy comparison
- Shows all 5 folds for each metric
- Clear labeling with units

**Insights:**
- Time-weighting improves ALL metrics
- Improvements are consistent across different evaluation approaches
- Validates model robustness

### 3. Aggregate Summary (`aggregate_summary.png`)

**Purpose:** Overall performance with uncertainty quantification

**Key Features:**
- Mean ± standard deviation for each metric
- Error bars showing variance
- Improvement annotations (green ▲ or red ▼)
- Scaled for visual comparison

**Insights:**
- Time-weighted model has LOWER variance (more stable)
- Baseline std: 5.11%, Time-weighted std: 4.59%
- More predictable, reliable performance

### 4. Improvement Heatmap (`improvement_heatmap.png`)

**Purpose:** Visual matrix of where time-weighting helps most

**Key Features:**
- Color-coded (green = improvement, red = regression)
- Normalized units for comparison
- Fold-by-fold breakdown

**Insights:**
- Most improvement in Fold 2 (2014-15)
- Fold 4 shows slight regression in accuracy
- Log-likelihood improvements are consistent

## Markdown Table

The script also generates a formatted markdown table for the README:

```markdown
| Fold | Period | Baseline Acc | Time-Weighted Acc | Improvement |
|------|--------|--------------|-------------------|-------------|
| 1 | 2013-2014 | 54.7% | **55.5%** | +0.8% |
| 2 | 2014-2015 | 51.6% | **52.6%** | +1.1% |
| 3 | 2015-2016 | 43.9% | **44.2%** | +0.3% |
| 4 | 2016-2017 | **57.6%** | 55.3% | -2.4% |
| 5 | 2017-2018 | 51.3% | **52.1%** | +0.8% |
```

**Features:**
- Bold highlighting for better model
- Clear improvement calculations
- Period labels for context

## Requirements

The visualization script requires:

```bash
pip install pandas matplotlib seaborn
```

These are already in `requirements.txt`, so if you've run `pip install -r requirements.txt`, you're good to go!

## Customization

### Change Chart Style

Edit the style settings at the top of `visualize_results.py`:

```python
sns.set_style("whitegrid")  # Options: whitegrid, darkgrid, white, dark, ticks
sns.set_palette("Set2")     # Options: Set1, Set2, Set3, Paired, etc.
plt.rcParams['figure.figsize'] = (12, 6)  # Adjust size
```

### Add More Charts

The script is modular. To add a new chart:

1. Create a function: `def plot_your_chart(df, output_dir):`
2. Generate your visualization
3. Save: `plt.savefig(output_dir / 'your_chart.png', dpi=300)`
4. Call in `main()`: `plot_your_chart(df, output_dir)`

### Change Chart Resolution

Current: 300 DPI (publication quality)

To change:
```python
plt.savefig(output_path, dpi=150)  # Lower (smaller file)
plt.savefig(output_path, dpi=600)  # Higher (larger file)
```

## Integration with README

The charts are referenced in `README.md` with relative paths:

```markdown
![Accuracy by Fold](data/backtest/charts/accuracy_by_fold.png)
```

**Important:** When viewing on GitHub:
- Charts display automatically in README
- Paths are relative to repository root
- PNG format ensures compatibility

## Updating Charts

After running new backtesting experiments:

1. Run backtesting: `python3 src/backtest/backtest_models.py`
2. Generate charts: `python3 src/backtest/visualize_results.py`
3. Charts automatically use latest results (sorted by timestamp)
4. No manual file selection needed!

## Troubleshooting

**"No backtesting results found"**
- Run `python3 src/backtest/backtest_models.py` first
- Check `data/backtest/` directory exists

**Charts look ugly/overlapping text**
- Increase figure size: `plt.rcParams['figure.figsize'] = (14, 8)`
- Adjust font size: `plt.rcParams['font.size'] = 10`

**Want different colors?**
- Change color codes in individual plot functions
- Use hex colors: `color='#e74c3c'` (red), `color='#2ecc71'` (green)

## Best Practices

1. **Always regenerate after backtesting** - Ensures charts match latest results
2. **Keep charts in repository** - GitHub displays them in README
3. **Use consistent styling** - All charts should look cohesive
4. **Add meaningful titles** - Help readers understand what they're seeing
5. **Include units** - Label axes clearly (%, ×100, etc.)

## Citation

If using these visualizations in academic work:

```bibtex
@software{epl_predictor_viz,
  title = {EPL Match Predictor - Backtesting Visualizations},
  author = {Rayner Goh},
  year = {2025},
  url = {https://github.com/raynergoh/EPL-Predictor}
}
```

---

**Questions?** Open an issue on GitHub: https://github.com/raynergoh/EPL-Predictor/issues
