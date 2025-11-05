"""
Generate visualization charts from backtesting results.

Creates professional charts for README documentation showing:
- Accuracy comparison across folds
- Performance metrics (Brier, Log-Likelihood)
- Model comparison summary

Usage:
    python3 src/backtest/visualize_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_latest_backtest_results():
    """Load the most recent backtesting results."""
    backtest_dir = Path("data/backtest")
    csv_files = sorted(backtest_dir.glob("backtest_results_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No backtesting results found. Run backtest_models.py first.")
    
    latest_file = csv_files[-1]
    print(f"📊 Loading results from: {latest_file.name}")
    
    df = pd.read_csv(latest_file)
    return df, latest_file.stem

def plot_accuracy_by_fold(df, output_dir):
    """Plot accuracy comparison across folds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    folds = df['fold'].unique()
    baseline_acc = df[df['model'] == 'Baseline (No Weighting)']['accuracy'].values * 100
    weighted_acc = df[df['model'].str.contains('Time-Weighted')]['accuracy'].values * 100
    
    x = np.arange(len(folds))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, baseline_acc, width, label='Baseline (No Weighting)', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, weighted_acc, width, label='Time-Weighted (ξ=0.003)', 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Test Fold (Season)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Model Accuracy by Test Fold (Walk-Forward Validation)', 
                fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in folds])
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 65)
    
    # Add horizontal line for average
    avg_baseline = baseline_acc.mean()
    avg_weighted = weighted_acc.mean()
    ax.axhline(y=avg_baseline, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=avg_weighted, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1.5)
    
    plt.tight_layout()
    output_path = output_dir / 'accuracy_by_fold.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_metrics_comparison(df, output_dir):
    """Plot all three metrics side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = [
        ('accuracy', 'Accuracy (%)', 100, True),  # (column, label, multiplier, higher_better)
        ('brier_score', 'Brier Score', 1, False),
        ('log_likelihood', 'Log-Likelihood', 1, True)
    ]
    
    for idx, (metric, label, multiplier, higher_better) in enumerate(metrics):
        ax = axes[idx]
        
        # Get data
        baseline_data = df[df['model'] == 'Baseline (No Weighting)'][metric].values * multiplier
        weighted_data = df[df['model'].str.contains('Time-Weighted')][metric].values * multiplier
        
        x = np.arange(len(df['fold'].unique()))
        width = 0.35
        
        # Colors based on improvement
        baseline_color = '#95a5a6'
        weighted_color = '#3498db'
        
        bars1 = ax.bar(x - width/2, baseline_data, width, label='Baseline', 
                      color=baseline_color, alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, weighted_data, width, label='Time-Weighted', 
                      color=weighted_color, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if metric == 'accuracy':
                    label_text = f'{height:.1f}%'
                else:
                    label_text = f'{height:.3f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label_text, ha='center', va='bottom' if height > 0 else 'top', 
                       fontsize=9, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Test Fold', fontweight='bold')
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(f'{label.split(" ")[0]} Comparison', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{i}' for i in df['fold'].unique()])
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_aggregate_summary(df, output_dir):
    """Plot aggregate performance summary."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate means and std
    baseline_df = df[df['model'] == 'Baseline (No Weighting)']
    weighted_df = df[df['model'].str.contains('Time-Weighted')]
    
    metrics_data = {
        'Accuracy (%)': (
            baseline_df['accuracy'].mean() * 100,
            weighted_df['accuracy'].mean() * 100,
            baseline_df['accuracy'].std() * 100,
            weighted_df['accuracy'].std() * 100
        ),
        'Brier Score\n(×100)': (
            baseline_df['brier_score'].mean() * 100,
            weighted_df['brier_score'].mean() * 100,
            baseline_df['brier_score'].std() * 100,
            weighted_df['brier_score'].std() * 100
        ),
        'Log-Likelihood\n(×10)': (
            baseline_df['log_likelihood'].mean() * 10,
            weighted_df['log_likelihood'].mean() * 10,
            baseline_df['log_likelihood'].std() * 10,
            weighted_df['log_likelihood'].std() * 10
        )
    }
    
    x = np.arange(len(metrics_data))
    width = 0.35
    
    baseline_means = [v[0] for v in metrics_data.values()]
    weighted_means = [v[1] for v in metrics_data.values()]
    baseline_stds = [v[2] for v in metrics_data.values()]
    weighted_stds = [v[3] for v in metrics_data.values()]
    
    # Create bars with error bars
    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                   label='Baseline (No Weighting)', color='#e67e22', alpha=0.8, 
                   edgecolor='black', capsize=5, error_kw={'linewidth': 2})
    bars2 = ax.bar(x + width/2, weighted_means, width, yerr=weighted_stds,
                   label='Time-Weighted (ξ=0.003)', color='#9b59b6', alpha=0.8, 
                   edgecolor='black', capsize=5, error_kw={'linewidth': 2})
    
    # Add value labels
    for bars, means, stds in [(bars1, baseline_means, baseline_stds), 
                               (bars2, weighted_means, weighted_stds)]:
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                   f'{mean:.2f}\n±{std:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('Performance (Scaled for Comparison)', fontweight='bold')
    ax.set_title('Average Performance Across All Test Folds\n(with Standard Deviation)', 
                fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_data.keys())
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add improvement annotations
    for idx, (metric, (b_mean, w_mean, _, _)) in enumerate(metrics_data.items()):
        improvement = w_mean - b_mean
        # For Brier score, lower is better
        if 'Brier' in metric:
            improvement = -improvement
        
        y_pos = max(b_mean, w_mean) + 3
        color = '#2ecc71' if improvement > 0 else '#e74c3c'
        symbol = '▲' if improvement > 0 else '▼'
        ax.text(idx, y_pos, f'{symbol} {abs(improvement):.2f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
    
    plt.tight_layout()
    output_path = output_dir / 'aggregate_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_improvement_heatmap(df, output_dir):
    """Plot heatmap showing improvement across metrics and folds."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Calculate improvements
    baseline_df = df[df['model'] == 'Baseline (No Weighting)'].sort_values('fold')
    weighted_df = df[df['model'].str.contains('Time-Weighted')].sort_values('fold')
    
    improvements = pd.DataFrame({
        'Fold': baseline_df['fold'].values,
        'Accuracy': (weighted_df['accuracy'].values - baseline_df['accuracy'].values) * 100,
        'Brier Score': (baseline_df['brier_score'].values - weighted_df['brier_score'].values) * 1000,  # Lower is better
        'Log-Likelihood': (weighted_df['log_likelihood'].values - baseline_df['log_likelihood'].values) * 100
    })
    
    improvements_matrix = improvements.set_index('Fold').T
    
    # Create heatmap
    sns.heatmap(improvements_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
               cbar_kws={'label': 'Improvement (Time-Weighted vs Baseline)'}, 
               linewidths=2, linecolor='black', ax=ax)
    
    ax.set_title('Time-Weighted Model Improvement by Fold and Metric\n(Positive = Better)', 
                fontweight='bold', pad=20)
    ax.set_xlabel('Test Fold', fontweight='bold')
    ax.set_ylabel('Metric', fontweight='bold')
    
    # Add units to y-labels
    yticklabels = ['Accuracy (pp)', 'Brier (×1000)', 'Log-Likelihood (×100)']
    ax.set_yticklabels(yticklabels, rotation=0)
    
    plt.tight_layout()
    output_path = output_dir / 'improvement_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def generate_markdown_table(df):
    """Generate markdown table for README."""
    print("\n" + "="*80)
    print("MARKDOWN TABLE FOR README")
    print("="*80 + "\n")
    
    baseline_df = df[df['model'] == 'Baseline (No Weighting)'].sort_values('fold')
    weighted_df = df[df['model'].str.contains('Time-Weighted')].sort_values('fold')
    
    print("| Fold | Period | Baseline Acc | Time-Weighted Acc | Improvement |")
    print("|------|--------|--------------|-------------------|-------------|")
    
    for idx, row in baseline_df.iterrows():
        fold = int(row['fold'])
        start_year = row['test_start_date'][:4]
        end_year = row['test_end_date'][:4]
        period = f"{start_year}-{end_year}"
        
        baseline_acc = row['accuracy'] * 100
        weighted_acc = weighted_df[weighted_df['fold'] == fold]['accuracy'].values[0] * 100
        improvement = weighted_acc - baseline_acc
        
        # Format with better/worse indicator
        if improvement > 0:
            better = "**" + f"{weighted_acc:.1f}%" + "**"
            improve_str = f"+{improvement:.1f}%"
        else:
            better = f"{weighted_acc:.1f}%"
            improve_str = f"{improvement:.1f}%"
        
        if baseline_acc > weighted_acc:
            baseline_str = "**" + f"{baseline_acc:.1f}%" + "**"
        else:
            baseline_str = f"{baseline_acc:.1f}%"
        
        print(f"| {fold} | {period} | {baseline_str} | {better} | {improve_str} |")
    
    print("\n" + "="*80 + "\n")

def main():
    """Generate all visualization charts."""
    print("\n" + "="*80)
    print("BACKTESTING RESULTS VISUALIZATION")
    print("="*80 + "\n")
    
    # Load data
    df, results_name = load_latest_backtest_results()
    
    # Create output directory
    output_dir = Path("data/backtest/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    # Generate all charts
    print("🎨 Generating charts...\n")
    plot_accuracy_by_fold(df, output_dir)
    plot_metrics_comparison(df, output_dir)
    plot_aggregate_summary(df, output_dir)
    plot_improvement_heatmap(df, output_dir)
    
    # Generate markdown table
    generate_markdown_table(df)
    
    print("\n" + "="*80)
    print("✅ ALL CHARTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📂 Charts saved to: {output_dir}/")
    print("📝 Copy the markdown table above to your README.md\n")

if __name__ == "__main__":
    main()
