"""
Analyze and visualize backtesting results.

Generates charts and detailed analysis of model performance across seasons.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_latest_results(results_dir: str = 'data/processed/backtest_results') -> Dict:
    """Load the most recent backtest results."""
    results_path = Path(results_dir)
    
    # Find latest results file
    result_files = sorted(results_path.glob('backtest_results_*.json'))
    if not result_files:
        raise FileNotFoundError("No backtest results found")
    
    latest_file = result_files[-1]
    print(f"Loading results from: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results


def create_season_accuracy_chart(results: Dict, output_dir: Path):
    """Create bar chart of accuracy by season."""
    season_data = pd.DataFrame([
        {
            'Season': r['season'],
            'Accuracy': r['accuracy'] * 100,
            'Matches': r['total_matches']
        }
        for r in results['season_results']
    ])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars = ax.bar(
        season_data['Season'], 
        season_data['Accuracy'],
        color='steelblue',
        edgecolor='navy',
        alpha=0.7
    )
    
    # Add baseline
    baseline = results['baseline_accuracy'] * 100
    ax.axhline(
        y=baseline, 
        color='red', 
        linestyle='--', 
        label=f'Random Baseline ({baseline:.1f}%)',
        linewidth=2
    )
    
    # Add overall accuracy line
    overall = results['overall_accuracy'] * 100
    ax.axhline(
        y=overall,
        color='green',
        linestyle='--',
        label=f'Overall Accuracy ({overall:.1f}%)',
        linewidth=2
    )
    
    # Styling
    ax.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy by Season', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{height:.1f}%',
            ha='center', 
            va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    output_file = output_dir / 'accuracy_by_season.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_outcome_metrics_chart(results: Dict, output_dir: Path):
    """Create grouped bar chart of precision/recall by outcome."""
    metrics = results['aggregate_outcome_metrics']
    
    outcomes = ['home_win', 'draw', 'away_win']
    outcome_labels = ['Home Win', 'Draw', 'Away Win']
    
    precision = [metrics[o]['precision'] * 100 for o in outcomes]
    recall = [metrics[o]['recall'] * 100 for o in outcomes]
    f1 = [metrics[o]['f1_score'] * 100 for o in outcomes]
    
    x = np.arange(len(outcomes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2E86AB')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#A23B72')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#F18F01')
    
    ax.set_xlabel('Outcome Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance by Outcome Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(outcome_labels)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 1:  # Only label if visible
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
    
    plt.tight_layout()
    output_file = output_dir / 'outcome_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_outcome_distribution_chart(results: Dict, output_dir: Path):
    """Create stacked bar chart showing outcome distribution per season."""
    season_outcomes = []
    
    for r in results['season_results']:
        dist = r['outcome_distribution']
        total = sum(dist.values())
        season_outcomes.append({
            'Season': r['season'],
            'Home Win': dist['home_win'] / total * 100,
            'Draw': dist['draw'] / total * 100,
            'Away Win': dist['away_win'] / total * 100
        })
    
    df = pd.DataFrame(season_outcomes)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df.plot(
        x='Season',
        kind='bar',
        stacked=True,
        ax=ax,
        color=['#2E86AB', '#A23B72', '#F18F01'],
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distribution (%)', fontsize=12, fontweight='bold')
    ax.set_title('Match Outcome Distribution by Season', fontsize=14, fontweight='bold')
    ax.legend(title='Outcome', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_file = output_dir / 'outcome_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_calibration_chart(results: Dict, output_dir: Path):
    """Create calibration plot (predicted probability vs actual frequency)."""
    # Aggregate calibration data across seasons
    all_calibration = {}
    
    for season_result in results['season_results']:
        if 'calibration' not in season_result:
            continue
        
        for bin_label, data in season_result['calibration'].items():
            if bin_label not in all_calibration:
                all_calibration[bin_label] = {
                    'predicted': [],
                    'actual': [],
                    'count': 0
                }
            
            all_calibration[bin_label]['predicted'].append(
                data['mean_predicted'] * data['count']
            )
            all_calibration[bin_label]['actual'].append(
                data['actual_frequency'] * data['count']
            )
            all_calibration[bin_label]['count'] += data['count']
    
    # Calculate weighted averages
    calibration_plot_data = []
    for bin_label in ['0-40%', '40-50%', '50-60%', '60-100%']:
        if bin_label in all_calibration:
            data = all_calibration[bin_label]
            total_count = data['count']
            
            if total_count > 0:
                avg_predicted = sum(data['predicted']) / total_count
                avg_actual = sum(data['actual']) / total_count
                
                calibration_plot_data.append({
                    'bin': bin_label,
                    'predicted': avg_predicted * 100,
                    'actual': avg_actual * 100,
                    'count': total_count
                })
    
    if calibration_plot_data:
        df = pd.DataFrame(calibration_plot_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot perfect calibration line
        ax.plot([0, 100], [0, 100], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Plot actual calibration
        ax.plot(
            df['predicted'], 
            df['actual'],
            'o-',
            color='steelblue',
            markersize=10,
            linewidth=2,
            label='Model Calibration'
        )
        
        # Add point labels
        for _, row in df.iterrows():
            ax.annotate(
                f"{row['bin']}\n(n={row['count']})",
                (row['predicted'], row['actual']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9
            )
        
        ax.set_xlabel('Predicted Probability (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Frequency (%)', fontsize=12, fontweight='bold')
        ax.set_title('Calibration Plot: Predicted vs Actual', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        output_file = output_dir / 'calibration_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def print_summary_report(results: Dict):
    """Print detailed text summary of results."""
    print("\n" + "="*70)
    print("DETAILED BACKTEST ANALYSIS")
    print("="*70)
    
    print(f"\n📊 OVERALL PERFORMANCE")
    print("-" * 70)
    print(f"Total Matches Tested: {results['total_matches']:,}")
    print(f"Correct Predictions: {results['total_correct']:,}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"Random Baseline: {results['baseline_accuracy']:.2%}")
    print(f"Improvement vs Baseline: +{results['improvement_vs_baseline']:.2%}")
    
    print(f"\n📈 OUTCOME-SPECIFIC METRICS")
    print("-" * 70)
    
    for outcome in ['home_win', 'draw', 'away_win']:
        metrics = results['aggregate_outcome_metrics'][outcome]
        print(f"\n{outcome.upper().replace('_', ' ')}:")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1 Score: {metrics['f1_score']:.2%}")
        print(f"  Actual Count: {metrics['total_count']:,}")
        print(f"  Predicted Count: {metrics['predicted_count']:,}")
        print(f"  Correct Count: {metrics['correct_count']:,}")
    
    print(f"\n🏆 BEST & WORST SEASONS")
    print("-" * 70)
    
    season_data = sorted(
        results['season_results'],
        key=lambda x: x['accuracy'],
        reverse=True
    )
    
    print("\nTop 3 Seasons:")
    for i, season in enumerate(season_data[:3], 1):
        print(f"  {i}. {season['season']}: {season['accuracy']:.2%} "
              f"({season['correct_predictions']}/{season['total_matches']})")
    
    print("\nBottom 3 Seasons:")
    for i, season in enumerate(season_data[-3:], 1):
        print(f"  {i}. {season['season']}: {season['accuracy']:.2%} "
              f"({season['correct_predictions']}/{season['total_matches']})")
    
    print(f"\n💡 KEY INSIGHTS")
    print("-" * 70)
    
    # Home advantage bias
    home_metrics = results['aggregate_outcome_metrics']['home_win']
    print(f"• Model has strong home win bias:")
    print(f"  - Predicts home wins {home_metrics['predicted_count']:,} times")
    print(f"  - Actual home wins: {home_metrics['total_count']:,} times")
    print(f"  - Home win recall: {home_metrics['recall']:.2%} (catches most home wins)")
    
    # Draw prediction challenge
    draw_metrics = results['aggregate_outcome_metrics']['draw']
    print(f"\n• Model struggles with draws:")
    print(f"  - Predicts draws only {draw_metrics['predicted_count']} time(s)")
    print(f"  - Actual draws: {draw_metrics['total_count']:,} times")
    print(f"  - Draw recall: {draw_metrics['recall']:.2%}")
    
    # Away win performance
    away_metrics = results['aggregate_outcome_metrics']['away_win']
    print(f"\n• Away win performance:")
    print(f"  - Balanced precision ({away_metrics['precision']:.2%}) "
          f"and recall ({away_metrics['recall']:.2%})")
    print(f"  - F1 Score: {away_metrics['f1_score']:.2%}")
    
    print("\n" + "="*70)


def main():
    """Generate all analysis and visualizations."""
    # Load results
    results = load_latest_results()
    
    # Create output directory
    output_dir = Path('data/processed/backtest_results/charts')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("GENERATING BACKTEST VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Generate charts
    create_season_accuracy_chart(results, output_dir)
    create_outcome_metrics_chart(results, output_dir)
    create_outcome_distribution_chart(results, output_dir)
    create_calibration_chart(results, output_dir)
    
    print(f"\n✅ All charts saved to: {output_dir}/")
    
    # Print summary report
    print_summary_report(results)
    
    print(f"\n📂 Full results available at:")
    print(f"   data/processed/backtest_results/")


if __name__ == '__main__':
    main()
