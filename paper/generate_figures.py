#!/usr/bin/env python3
"""
Generate publication-quality figures for the sparse quantum simulation paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def figure1_sparsity_evolution():
    """Sparsity evolution under different circuit classes."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n = np.array([10, 20, 30, 50, 100, 200, 500, 1000])

    ghz_sparsity = np.full_like(n, 2, dtype=float)
    w_sparsity = n.astype(float)
    dicke_sparsity = (n * (n - 1) / 2).astype(float)
    exp_sparsity = np.minimum(2.0 ** n, 1e12)

    ax.semilogy(n, ghz_sparsity, 'b-o', linewidth=2, markersize=6, label='GHZ (Class I): 2')
    ax.semilogy(n, w_sparsity, 'g-s', linewidth=2, markersize=6, label='W-state (Class I): n')
    ax.semilogy(n, dicke_sparsity, 'orange', linestyle='-', marker='^', linewidth=2, markersize=6, label=r'Dicke (Class II): C(n,2)')
    ax.semilogy(n, exp_sparsity, 'r-d', linewidth=2, markersize=6, label=r'Graph state (Class III): $2^n$')

    ax.axhspan(1, 1e4, alpha=0.1, color='green')
    ax.axhspan(1e4, 1e8, alpha=0.1, color='yellow')
    ax.axhspan(1e8, 1e15, alpha=0.1, color='red')

    ax.set_xlabel('Number of Qubits (n)')
    ax.set_ylabel('Number of Non-zero Amplitudes')
    ax.set_title('Sparsity Evolution by Circuit Class')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1050)
    ax.set_ylim(1, 1e13)

    ax.annotate('Efficient', xy=(800, 100), fontsize=10, color='green')
    ax.annotate('Tractable', xy=(800, 1e6), fontsize=10, color='orange')
    ax.annotate('Infeasible', xy=(800, 1e10), fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('figure1_sparsity_evolution.png')
    print("Generated: figure1_sparsity_evolution.png")
    plt.close()

def figure2_method_comparison():
    """Comparison of efficient simulation methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ['Block-Sparse\n(This Work)', 'Stabilizer\n(Gottesman-Knill)', 'MPS/Tensor\nNetworks']
    states = ['GHZ', 'W-state', 'Dicke', 'Graph State', '1D Cluster', 'Random']

    efficiency = np.array([
        [2, 2, 2, 0, 0, 0],
        [2, 0, 0, 2, 2, 0],
        [2, 1, 1, 1, 2, 1],
    ])

    cmap = plt.cm.RdYlGn
    im = ax.imshow(efficiency, cmap=cmap, aspect='auto', vmin=0, vmax=2)

    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, rotation=45, ha='right')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)

    for i in range(len(methods)):
        for j in range(len(states)):
            val = efficiency[i, j]
            text = ['Exp', 'Poly', 'O(k)'][val]
            color = 'white' if val == 0 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9, fontweight='bold')

    ax.set_title('Simulation Method Efficiency Comparison')

    legend_elements = [
        mpatches.Patch(facecolor=cmap(1.0), label='Efficient O(k) or O(n)'),
        mpatches.Patch(facecolor=cmap(0.5), label='Tractable poly(n)'),
        mpatches.Patch(facecolor=cmap(0.0), label='Inefficient exp(n)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1))

    plt.tight_layout()
    plt.savefig('figure2_method_comparison.png', bbox_inches='tight')
    print("Generated: figure2_method_comparison.png")
    plt.close()

def figure3_scaling():
    """Memory and time scaling comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    n = np.array([10, 20, 30, 40, 50, 100, 500, 1000])
    sparse_mem = np.full_like(n, 4, dtype=float)
    dense_mem = 16 * (2.0 ** n) / 1024
    mps_mem = 4 * 8 * n / 1024

    ax1.semilogy(n, sparse_mem, 'b-o', linewidth=2, markersize=6, label='Block-Sparse (This Work)')
    ax1.semilogy(n[:5], dense_mem[:5], 'r--s', linewidth=2, markersize=6, label='Dense')
    ax1.semilogy(n, mps_mem, 'g-.^', linewidth=2, markersize=6, label='MPS')

    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Memory (KB)')
    ax1.set_title('(a) Memory Scaling: GHZ State')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.1, 1e10)

    n_time = np.array([100, 500, 1000, 5000, 10000, 100000, 1000000])
    sparse_time = np.array([0.065, 0.236, 0.5, 4.3, 11.6, 516, 53000])

    ax2.loglog(n_time, sparse_time, 'b-o', linewidth=2, markersize=6, label='Block-Sparse')
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('(b) Time Scaling: GHZ State Creation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure3_scaling.png')
    print("Generated: figure3_scaling.png")
    plt.close()

if __name__ == "__main__":
    print("Generating figures...")
    figure1_sparsity_evolution()
    figure2_method_comparison()
    figure3_scaling()
    print("All figures generated!")
