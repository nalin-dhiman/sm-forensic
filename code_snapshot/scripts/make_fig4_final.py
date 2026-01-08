#!/usr/bin/env python3
"""
Final Generation Script for Full Figure 4 (Panels A, B, C).

Requirements:
1. N=20 seeds for statistics (Panel C).
2. Representative seed (Seed 0) for temporal/spectral plots (Panel A, B).
3. Panel A: Population Rate (Hz) vs Time (ms).
4. Panel B: PSD vs Freq (Hz).
5. Panel C: Paired Boxplot of Gamma Power.
6. Correct Direction: Expect Amplification (Emb > Iso).
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy import stats, signal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from smforensic.config import DEFAULT_ADEX, DEFAULT_NOISE, DEFAULT_SIM, DEFAULT_SYN, SimParams, SynapseParams
from smforensic.adex import AdExSimulator
from smforensic.assays import gamma_band_power
from smforensic.stats import paired_wilcoxon, cohens_dz

def get_psd(rate_hz, dt_ms):
    fs = 1000.0 / dt_ms
    f, Pxx = signal.welch(rate_hz, fs=fs, nperseg=2048)
    return f, Pxx

def main():
    # --- 1. Simulation Setup ---
    CONNECTOME_PATH = Path("data/connectome.npz")
    SEEDS = np.arange(20)
    GAMMA_BAND = (30.0, 80.0)
    ALPHA = 6.0 
    
    print(f"Loading connectome from {CONNECTOME_PATH}...")
    with np.load(CONNECTOME_PATH) as data:
        W_counts = data['W_counts']
        pre_signs = data['pre_signs']
    
    W_dense = W_counts * pre_signs[None, :]
    W_emb = sp.csr_matrix(W_dense)
    N_neurons = W_emb.shape[0]
    W_iso = sp.csr_matrix((N_neurons, N_neurons), dtype=float)
    
    syn_params = SynapseParams(tau_syn_ms=DEFAULT_SYN.tau_syn_ms, alpha_pA_per_count=ALPHA)
    
    print(f"Running N={len(SEEDS)} simulations...")
    
    gamma_iso = []
    gamma_emb = []
    
    # Store rep data for Seed 0
    rep_time = None
    rep_iso_rate = None
    rep_emb_rate = None
    
    for seed in tqdm(SEEDS):
        sim_params = SimParams(
            dt_ms=DEFAULT_SIM.dt_ms,
            T_ms=2000.0,
            Ibias_pA=DEFAULT_SIM.Ibias_pA,
            seed=int(seed),
            min_rate_hz=DEFAULT_SIM.min_rate_hz,
            max_rate_hz=DEFAULT_SIM.max_rate_hz
        )
        burn_steps = int(200.0 / sim_params.dt_ms)
        
        # Iso
        simr_iso = AdExSimulator(W_iso, adex=DEFAULT_ADEX, syn=syn_params, noise=DEFAULT_NOISE, sim=sim_params)
        res_iso = simr_iso.run(record_spikes=False)
        rate_iso = res_iso.rate_hz
        _, _, p_iso = gamma_band_power(rate_iso[burn_steps:], dt_ms=sim_params.dt_ms, band_hz=GAMMA_BAND)
        gamma_iso.append(p_iso)
        
        # Emb
        simr_emb = AdExSimulator(W_emb, adex=DEFAULT_ADEX, syn=syn_params, noise=DEFAULT_NOISE, sim=sim_params)
        res_emb = simr_emb.run(record_spikes=False)
        rate_emb = res_emb.rate_hz
        _, _, p_emb = gamma_band_power(rate_emb[burn_steps:], dt_ms=sim_params.dt_ms, band_hz=GAMMA_BAND)
        gamma_emb.append(p_emb)
        
        # Capture Rep Data (Seed 0)
        if int(seed) == 0:
            rep_time = res_iso.t_ms
            rep_iso_rate = rate_iso
            rep_emb_rate = rate_emb

    gamma_iso = np.array(gamma_iso)
    gamma_emb = np.array(gamma_emb)
    
    # --- 3. Statistics ---
    stat, p_val = paired_wilcoxon(gamma_emb, gamma_iso)
    dz = cohens_dz(gamma_emb, gamma_iso)
    
    print(f"\nWilcoxon p-value (N={len(SEEDS)}): {p_val:.4e}")
    print(f"Effect Size dz: {dz:.4f}")

    # --- 4. Plotting ---
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    
    # Colors
    c_iso = '#ff4d4d' # Red
    c_emb = 'black'
    c_iso_fill = '#ffcccc' # Pink for box
    c_emb_fill = '#bbbbbb' # Gray for box

    # PANEL A: Rate vs Time
    ax = axes[0]
    # Smooth for display
    sigma_ms = 10.0
    sigma_steps = sigma_ms / 0.1 # dt=0.1
    iso_smooth = gaussian_filter1d(rep_iso_rate, sigma_steps)
    emb_smooth = gaussian_filter1d(rep_emb_rate, sigma_steps)
    
    # Plot window 500-1500ms
    mask = (rep_time >= 500) & (rep_time <= 1500)
    t_plot = rep_time[mask]
    
    ax.plot(t_plot, emb_smooth[mask], color=c_emb, label='Embedded')
    ax.plot(t_plot, iso_smooth[mask], color=c_iso, label='Isolated')
    
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_title("A", loc='left', fontweight='bold', fontsize=12)
    ax.legend(frameon=False, loc='center right')
    
    # PANEL B: PSD
    ax = axes[1]
    # Compute PSD on steady state (200ms+)
    f_iso, P_iso = get_psd(rep_iso_rate[2000:], 0.1)
    f_emb, P_emb = get_psd(rep_emb_rate[2000:], 0.1)
    
    ax.plot(f_emb, P_emb, color=c_emb)
    ax.plot(f_iso, P_iso, color=c_iso)
    
    ax.set_xlim(0, 100)
    # Highlight Gamma band
    ax.axvspan(30, 80, color='yellow', alpha=0.1, zorder=0)
    
    ax.set_xlabel("Freq (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("B", loc='left', fontweight='bold', fontsize=12)
    
    # PANEL C: Boxplot
    ax = axes[2]
    data = [gamma_iso, gamma_emb]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True, 
                    showfliers=False, labels=['Iso', 'Emb'])
    
    bp['boxes'][0].set_facecolor(c_iso_fill)
    bp['boxes'][1].set_facecolor(c_emb_fill)
    for patch in bp['boxes']:
        patch.set_edgecolor('black')
        
    for median in bp['medians']:
        median.set_color('orange')
        
    # Connecting Lines
    for i in range(len(gamma_iso)):
        ax.plot([0, 1], [gamma_iso[i], gamma_emb[i]], color='gray', alpha=0.3, linewidth=0.8)
        
    ax.set_ylabel("Gamma Power (30-80Hz)")
    ax.set_title("C", loc='left', fontweight='bold', fontsize=12)
    
    # Add stats title to C (centered)
    ax.text(0.5, 1.05, f"Amplification\np = {p_val:.1e}, dz={dz:.2f}", 
            transform=ax.transAxes, ha='center', va='bottom', fontsize=10)

    # Save
    out_path = Path("paper/figures/Fig4_gamma_reproduction_full.png")
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path.with_suffix(".pdf"))
    print(f"Saved full figure to {out_path} and .pdf")

if __name__ == "__main__":
    main()
