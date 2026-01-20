#!/usr/bin/env python3
"""Create schematic diagram for dual-scale model"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'DUAL-SCALE ARCHITECTURE OF Q47', fontsize=18, 
        ha='center', fontweight='bold', color='navy')

# === LEFT SIDE: Poisson Background ===
# Box
box1 = FancyBboxPatch((0.5, 1.5), 5, 6.5, boxstyle="round,pad=0.1",
                       facecolor='#e8f4e8', edgecolor='darkgreen', linewidth=3)
ax.add_patch(box1)

# Title
ax.text(3, 7.5, 'BACKGROUND', fontsize=14, ha='center', fontweight='bold', color='darkgreen')
ax.text(3, 6.8, '(Scale 1)', fontsize=11, ha='center', color='darkgreen')

# Random dots (Poisson)
np.random.seed(42)
n_dots = 80
x_dots = np.random.uniform(1, 5, n_dots)
y_dots = np.random.uniform(2, 6, n_dots)
ax.scatter(x_dots, y_dots, c='red', s=20, alpha=0.6, zorder=5)

# Labels
ax.text(3, 1.8, '8.9 Million Primes', fontsize=12, ha='center', fontweight='bold')
ax.text(3, 1.3, 'Statistics: POISSON', fontsize=11, ha='center', color='green')
ax.text(3, 0.8, 'Δ₃ = 6.80 ≈ L/15', fontsize=10, ha='center')
ax.text(3, 0.3, '"Arithmetic Heat Bath"', fontsize=10, ha='center', style='italic')

# === RIGHT SIDE: GUE Condensate ===
# Box
box2 = FancyBboxPatch((8.5, 1.5), 5, 6.5, boxstyle="round,pad=0.1",
                       facecolor='#fff4e8', edgecolor='darkred', linewidth=3)
ax.add_patch(box2)

# Title
ax.text(11, 7.5, 'COHERENT STRUCTURES', fontsize=14, ha='center', fontweight='bold', color='darkred')
ax.text(11, 6.8, '(Scale 2)', fontsize=11, ha='center', color='darkred')

# Ordered quadruplets
quadruplet_y = [5.5, 4.5, 3.5, 2.5]
for i, y in enumerate(quadruplet_y):
    # Draw 4 aligned points (quadruplet)
    for j in range(4):
        c = Circle((9.5 + j*0.8, y), 0.15, facecolor='blue', edgecolor='darkblue', linewidth=2, zorder=5)
        ax.add_patch(c)
    # Connect them
    ax.plot([9.5, 9.5+2.4], [y, y], 'b-', lw=2, zorder=4)
    
# Labels
ax.text(11, 1.8, '15 Quadruplets', fontsize=12, ha='center', fontweight='bold')
ax.text(11, 1.3, 'Statistics: GUE-like', fontsize=11, ha='center', color='red')
ax.text(11, 0.8, 'r = 0.994 (Riemann)', fontsize=10, ha='center')
ax.text(11, 0.3, '"Arithmetic Solitons"', fontsize=10, ha='center', style='italic')

# === CENTRAL ARROW ===
arrow = FancyArrowPatch((5.8, 4.5), (8.2, 4.5),
                         arrowstyle='-|>', mutation_scale=30,
                         color='purple', lw=4)
ax.add_patch(arrow)
ax.text(7, 5.5, 'Selective\nCondensation', fontsize=12, ha='center', 
        fontweight='bold', color='purple')
ax.text(7, 3.3, 'λ ~ 10⁻⁶', fontsize=11, ha='center', color='purple')

# === BOTTOM BOX: Key Insight ===
box3 = FancyBboxPatch((1, -1.2), 12, 1.5, boxstyle="round,pad=0.1",
                       facecolor='#f0f0ff', edgecolor='navy', linewidth=2)
ax.add_patch(box3)
ax.text(7, -0.2, 'KEY INSIGHT: The Poisson background is not "failure to find GUE"', 
        fontsize=12, ha='center', fontweight='bold', color='navy')
ax.text(7, -0.7, '— it is the NECESSARY THERMAL RESERVOIR for condensate emergence', 
        fontsize=11, ha='center', color='navy')

# Equation
ax.text(7, 8.7, r'$R_2(r) \approx (1-\lambda) \cdot 1 + \lambda \cdot [1 - (\sin\pi r/\pi r)^2]$', 
        fontsize=12, ha='center', style='italic')

plt.tight_layout()
plt.savefig('fig_schematic_dualscale.pdf', bbox_inches='tight', facecolor='white')
plt.savefig('fig_schematic_dualscale.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Schematic saved")
