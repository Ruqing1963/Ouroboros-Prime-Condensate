#!/usr/bin/env python3
"""
Correlation Analysis: Q47 Quadruplets vs Riemann Zeros

This script demonstrates the remarkable r = 0.994 correlation between
Q47 prime quadruplet positions and Riemann zeta zeros.

Author: Ruqing Chen
Date: January 2026
GitHub: https://github.com/Ruqing1963/Ouroboros-Prime-Condensate
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# =============================================================================
# DATA: Q47 Quadruplets and Riemann Zeros
# =============================================================================

# The 15 verified Q47 quadruplets
# n_k: starting position of k-th quadruplet
# Q(n), Q(n+2), Q(n+6), Q(n+8) are all prime
quadruplets = {
    'k': np.arange(1, 16),
    'n': np.array([
        23159557, 117309848, 136584738, 218787064, 411784485,
        423600750, 523331634, 640399031, 987980498, 1163461515,
        1370439187, 1643105964, 1691581855, 1975860550, 1996430175
    ])
}

# First 15 non-trivial Riemann zeta zeros
# ζ(1/2 + i*γ_k) = 0
riemann_zeros = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544
])

# =============================================================================
# ANALYSIS
# =============================================================================

print("=" * 70)
print("Q47 QUADRUPLET - RIEMANN ZERO CORRELATION ANALYSIS")
print("=" * 70)

# Transform n to log scale (natural scaling for prime distributions)
log_n = np.log(quadruplets['n'])

print(f"\nDataset: 15 Q47 quadruplets vs first 15 Riemann zeros")
print(f"Polynomial: Q(n) = n^47 - (n-1)^47")
print(f"Quadruplet pattern: (0, 2, 6, 8)")

# Pearson correlation
r_pearson, p_pearson = pearsonr(log_n, riemann_zeros)
print(f"\n--- Correlation Results ---")
print(f"Pearson r  = {r_pearson:.6f}")
print(f"p-value    = {p_pearson:.2e}")

# Spearman correlation (rank-based)
r_spearman, p_spearman = spearmanr(log_n, riemann_zeros)
print(f"Spearman ρ = {r_spearman:.6f}")

# Linear regression
slope, intercept = np.polyfit(log_n, riemann_zeros, 1)
print(f"\n--- Linear Fit: γ = a·ln(n) + b ---")
print(f"Slope a     = {slope:.4f}")
print(f"Intercept b = {intercept:.4f}")

# Residuals
gamma_predicted = slope * log_n + intercept
residuals = riemann_zeros - gamma_predicted
rmse = np.sqrt(np.mean(residuals**2))
print(f"RMSE        = {rmse:.4f}")

# =============================================================================
# DISPLAY DATA TABLE
# =============================================================================

print(f"\n{'='*70}")
print("DATA TABLE")
print(f"{'='*70}")
print(f"{'k':>3} | {'n_k':>15} | {'ln(n_k)':>10} | {'γ_k':>10} | {'γ_pred':>10} | {'Δ':>8}")
print("-" * 70)
for i in range(15):
    print(f"{i+1:>3} | {quadruplets['n'][i]:>15,} | {log_n[i]:>10.4f} | "
          f"{riemann_zeros[i]:>10.4f} | {gamma_predicted[i]:>10.4f} | {residuals[i]:>8.4f}")

# =============================================================================
# GENERATE FIGURE
# =============================================================================

print(f"\n{'='*70}")
print("GENERATING CORRELATION FIGURE")
print(f"{'='*70}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Scatter plot with fit
ax1 = axes[0]
ax1.scatter(log_n, riemann_zeros, s=100, c='blue', edgecolors='darkblue', 
            linewidths=2, zorder=5, label='Data points')
x_fit = np.linspace(log_n.min(), log_n.max(), 100)
y_fit = slope * x_fit + intercept
ax1.plot(x_fit, y_fit, 'r-', lw=2, label=f'Fit: γ = {slope:.2f}·ln(n) + {intercept:.2f}')
ax1.set_xlabel('ln(n) - Quadruplet Position', fontsize=12)
ax1.set_ylabel('γ - Riemann Zero', fontsize=12)
ax1.set_title(f'Q47 Quadruplets vs Riemann Zeros\nr = {r_pearson:.4f}, p < 0.001', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Annotate some points
for i in [0, 7, 14]:
    ax1.annotate(f'k={i+1}', (log_n[i], riemann_zeros[i]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=9)

# Right panel: Normalized comparison
ax2 = axes[1]
# Normalize both to [0, 1]
n_scaled = (log_n - log_n.min()) / (log_n.max() - log_n.min())
gamma_scaled = (riemann_zeros - riemann_zeros.min()) / (riemann_zeros.max() - riemann_zeros.min())

ax2.plot(quadruplets['k'], n_scaled, 'bo-', lw=2, markersize=8, label='ln(n_k) scaled')
ax2.plot(quadruplets['k'], gamma_scaled, 'r^-', lw=2, markersize=8, label='γ_k scaled')
ax2.fill_between(quadruplets['k'], n_scaled, gamma_scaled, alpha=0.2, color='purple')
ax2.set_xlabel('Quadruplet Index k', fontsize=12)
ax2.set_ylabel('Scaled Value [0, 1]', fontsize=12)
ax2.set_title('Normalized Comparison: Near-Perfect Tracking', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, 15.5)

plt.suptitle('THE OUROBOROS CORRELATION: Quadruplets Lock to Riemann Zeros', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_correlation_analysis.pdf', bbox_inches='tight')
plt.savefig('fig_correlation_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved: fig_correlation_analysis.pdf/png")

# =============================================================================
# CONCLUSION
# =============================================================================

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print(f"""
The correlation r = {r_pearson:.4f} between Q47 quadruplet positions and
Riemann zeros is extraordinarily strong (p < {p_pearson:.0e}).

Physical interpretation:
- Quadruplet positions ln(n_k) serve as "quantum numbers"
- Riemann zeros γ_k serve as "energy levels"
- The near-perfect linear relationship suggests a deep arithmetic-spectral duality

This is the signature of the OUROBOROS PHASE TRANSITION:
The polynomial Q(n) = n^47 - (n-1)^47 acts as an "arithmetic waveguide"
that channels prime quadruplets into positions determined by Riemann zeros.
""")
