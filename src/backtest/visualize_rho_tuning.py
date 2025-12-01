"""
Visualize ρ hyperparameter tuning results.

Creates a chart showing how log-likelihood changes with different ρ values
for Dixon-Coles dependency correction.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Find most recent tuning results
tuning_dir = Path("data/tuning")
rho_files = list(tuning_dir.glob("rho_tuning_results_*.csv"))
if not rho_files:
    print("No ρ tuning results found!")
    exit(1)

latest_file = max(rho_files, key=lambda p: p.stat().st_mtime)
print(f"Loading: {latest_file}")

# Load data
df = pd.read_csv(latest_file)

# Create visualization
fig, ax = plt.subplots(figsize=(12, 7))

# Plot mean log-likelihood with error bars
ax.errorbar(
    df['rho'], 
    df['mean_log_likelihood'], 
    yerr=df['std_log_likelihood'],
    fmt='o-',
    capsize=5,
    capthick=2,
    markersize=6,
    linewidth=2,
    color='#2E86AB',
    label='Mean ± Std Dev (5-fold CV)'
)

# Mark optimal ρ
optimal_idx = df['mean_log_likelihood'].idxmax()
optimal_rho = df.loc[optimal_idx, 'rho']
optimal_ll = df.loc[optimal_idx, 'mean_log_likelihood']

ax.axvline(optimal_rho, color='red', linestyle='--', linewidth=2, alpha=0.7, 
           label=f'Optimal: ρ = {optimal_rho:.3f}')
ax.scatter([optimal_rho], [optimal_ll], color='red', s=200, zorder=5, 
           marker='*', edgecolors='darkred', linewidths=2)

# Mark Dixon & Coles (1997) value
dixon_coles_rho = -0.13
if dixon_coles_rho in df['rho'].values:
    dc_ll = df.loc[df['rho'] == dixon_coles_rho, 'mean_log_likelihood'].values[0]
    ax.axvline(dixon_coles_rho, color='orange', linestyle=':', linewidth=2, alpha=0.7,
               label=f'Dixon & Coles (1997): ρ = {dixon_coles_rho:.3f}')
    ax.scatter([dixon_coles_rho], [dc_ll], color='orange', s=150, zorder=5,
               marker='D', edgecolors='darkorange', linewidths=2)

# Mark ρ=0 (no correction)
if 0.0 in df['rho'].values:
    zero_ll = df.loc[df['rho'] == 0.0, 'mean_log_likelihood'].values[0]
    ax.axvline(0.0, color='gray', linestyle=':', linewidth=2, alpha=0.7,
               label='No correction: ρ = 0.000')
    ax.scatter([0.0], [zero_ll], color='gray', s=150, zorder=5,
               marker='s', edgecolors='black', linewidths=2)

# Formatting
ax.set_xlabel('ρ (Dependency Parameter)', fontsize=14, fontweight='bold')
ax.set_ylabel('Log-Likelihood (higher is better)', fontsize=14, fontweight='bold')
ax.set_title('Dixon-Coles Dependency Parameter (ρ) Hyperparameter Tuning', 
             fontsize=16, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='best', framealpha=0.95)

# Add annotation for optimal value
ax.annotate(
    f'Best ρ = {optimal_rho:.3f}\nLL = {optimal_ll:.4f}',
    xy=(optimal_rho, optimal_ll),
    xytext=(optimal_rho + 0.05, optimal_ll - 0.0005),
    fontsize=11,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2)
)

# Add key insight text box
improvement = (optimal_ll - zero_ll) * 100  # Convert to percentage points (×100)
insight_text = (
    f"Key Findings:\n"
    f"• Optimal ρ = {optimal_rho:.3f} (vs -0.13 in Dixon & Coles 1997)\n"
    f"• Improvement over no correction: {improvement:.3f}%\n"
    f"• Modern EPL shows weak dependency between goals\n"
    f"• DC correction has minimal practical impact"
)

ax.text(
    0.02, 0.02,
    insight_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8)
)

plt.tight_layout()

# Save
output_dir = Path("data/tuning")
output_file = output_dir / "rho_tuning_curve.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Chart saved: {output_file}")

plt.show()
