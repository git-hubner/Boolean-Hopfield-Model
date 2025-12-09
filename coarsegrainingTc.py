import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ==================== PARAMETRI ====================
N = 10000
f = 0.1
s_e = 0.7
s_t = np.sqrt(1 - s_e**2)
T_threshold = norm.ppf(1 - f)

# Parametri simulazione
theta = 0.6

# RANGE DI PARAMETRI DA ESPLORARE
lambda_values = np.linspace(0, 0.9, 10)  # 0, 0.1, 0.2, ..., 0.9
alpha_values = np.linspace(0.2, 0.6, 9)  # 0.2, 0.25, 0.3, ..., 0.6
n_realizations = 2  # Realizzazioni per ogni combinazione

# Parametri stati misti
patterns_per_mixed = 20  # Numero di pattern da combinare per ogni stato misto
sigma_gaussian = 10.0    # Sigma per pesi gaussiani

# PARAMETRI PER RICERCA TRANSIZIONE
T_initial_min = 0.001
T_initial_max = 0.5
n_sweep_mc = 30  # Sweep Monte Carlo per ogni temperatura

# Crea cartella output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"hierarchical_phase_transition_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print(f"HIERARCHICAL SCAN: PHASE TRANSITION DETECTION")
print("=" * 80)
print(f"λ values: {len(lambda_values)} points from {lambda_values[0]:.2f} to {lambda_values[-1]:.2f}")
print(f"α values: {len(alpha_values)} points from {alpha_values[0]:.2f} to {alpha_values[-1]:.2f}")
print(f"Realizzazioni per combinazione: {n_realizations}")
print(f"Mode: ADAPTIVE temperature search for phase transition")
print(f"Output directory: {output_dir}/")
print(f"Total runs: {len(lambda_values) * len(alpha_values) * n_realizations}")
print("=" * 80)

def prob(h, tht, beta):
    with np.errstate(over='ignore'):
        return 1/(1 + np.exp(-(h - tht) * beta))

def magn(Z, eta_p, f):
    return np.dot(eta_p - f, Z) / ((1 - f) * np.sum(eta_p))

def generate_patterns(P, N, lam, s_e, s_t, T_threshold):
    """Genera i pattern originali con correlazione spaziale"""
    eta = np.zeros((P, N))
    I_ep = np.random.normal(0, s_e, size=(P, N))
    I_t = np.zeros((P, N))
    
    if lam == 0:
        I_t = s_t * np.random.normal(size=(P, N))
    else:
        I_t[0] = s_t * np.sqrt(1 - lam**2) * np.random.normal(size=N)
        for i in range(1, P):
            z_t = np.random.normal(size=N)
            I_t[i] = lam * I_t[i-1] + s_t * np.sqrt(1 - lam**2) * z_t
    
    eta = (I_t + I_ep - T_threshold >= 0).astype(float)
    return eta

def build_J_matrix(eta, f, N):
    """Costruisce la matrice J standard"""
    P_eff = eta.shape[0]
    eta_centered = eta - f
    J = np.dot(eta_centered.T, eta_centered) / (f * (1 - f) * N)
    np.fill_diagonal(J, 0)
    return J

def make_weighted_mixed_pattern(eta, start, end, center=None, sigma=10.0, f=0.1):
    """Crea un singolo pattern misto da eta[start:end+1] con pesi gaussiani"""
    P, N = eta.shape
    pattern_indices = list(range(start, end+1))
    L = len(pattern_indices)
    if center is None:
        center = (start + end) / 2.0
    
    mus = np.array(pattern_indices)
    dist = mus - center
    w_raw = np.exp(-0.5 * (dist / sigma)**2)
    w = w_raw / w_raw.sum()
    
    S = np.zeros(N, dtype=float)
    for k, mu in enumerate(pattern_indices):
        S += w[k] * eta[mu]
    
    k = int(round(f * N))
    idx = np.argsort(S)
    sel = idx[-k:]
    M_bin = np.zeros(N, dtype=float)
    M_bin[sel] = 1.0
    
    return M_bin, w

def create_all_mixed_patterns(eta, patterns_per_mixed, f, sigma):
    """Crea P/patterns_per_mixed stati misti"""
    P, N = eta.shape
    n_mixed = P // patterns_per_mixed
    
    if n_mixed == 0:
        return None, None
    
    eta_new = np.zeros((n_mixed, N))
    
    for i in range(n_mixed):
        start = i * patterns_per_mixed
        end = start + patterns_per_mixed - 1
        center = (start + end) / 2.0
        
        M_bin, w = make_weighted_mixed_pattern(
            eta, start, end, center=center, sigma=sigma, f=f
        )
        
        eta_new[i] = M_bin
    
    return eta_new, n_mixed

def run_dynamics_at_temperature(W_init, J, target_pattern, f, theta, T, n_sweeps=30):
    """Esegue dinamica Monte Carlo a temperatura fissata"""
    W = W_init.copy()
    
    for _ in range(n_sweeps):
        for i in range(len(W)):
            h = np.dot(J[i], W)
            phi = prob(h, theta, 1/T)
            if np.random.rand() < phi:
                W[i] = 1
            else:
                W[i] = 0
    
    m_target = magn(W, target_pattern, f)
    q_0 = np.sum(W) / len(W)
    
    return m_target, q_0

def find_phase_transition(J, target_pattern, f, theta, 
                          T_min=0.001, T_max=0.5, 
                          threshold_high=0.7, threshold_low=0.3,
                          max_iterations=15, n_sweeps=30):
    """
    Ricerca adattiva della transizione di fase usando bisection.
    
    Returns:
    - T_c: temperatura critica (dove m_target ≈ 0.5)
    - m_high: magnetizzazione a T bassa (retrieval buono)
    - m_low: magnetizzazione a T alta (retrieval cattivo)
    - T_values: temperature testate
    - m_values: magnetizzazioni corrispondenti
    """
    
    # Stato iniziale con rumore
    W_init = target_pattern.copy()
    flip_idx = np.random.choice(N, size=N//30, replace=False)
    W_init[flip_idx] = 1 - W_init[flip_idx]
    
    T_values = []
    m_values = []
    
    # Test estremi
    m_low_temp, _ = run_dynamics_at_temperature(W_init, J, target_pattern, f, theta, T_min, n_sweeps)
    T_values.append(T_min)
    m_values.append(m_low_temp)
    
    m_high_temp, _ = run_dynamics_at_temperature(W_init, J, target_pattern, f, theta, T_max, n_sweeps)
    T_values.append(T_max)
    m_values.append(m_high_temp)
    
    # Se non c'è transizione, ritorna subito
    if m_low_temp < threshold_high or m_high_temp > threshold_low:
        # Nessuna transizione chiara
        return None, m_low_temp, m_high_temp, T_values, m_values
    
    # Bisection per trovare T_c
    T_left = T_min
    T_right = T_max
    m_left = m_low_temp
    m_right = m_high_temp
    
    for iteration in range(max_iterations):
        T_mid = (T_left + T_right) / 2
        m_mid, _ = run_dynamics_at_temperature(W_init, J, target_pattern, f, theta, T_mid, n_sweeps)
        
        T_values.append(T_mid)
        m_values.append(m_mid)
        
        # Check convergenza
        if abs(m_mid - 0.5) < 0.05 or (T_right - T_left) < 0.005:
            # Trovata transizione
            T_c = T_mid
            return T_c, m_left, m_right, T_values, m_values
        
        # Update bounds
        if m_mid > 0.5:
            T_left = T_mid
            m_left = m_mid
        else:
            T_right = T_mid
            m_right = m_mid
    
    # Max iterations raggiunto
    T_c = (T_left + T_right) / 2
    return T_c, m_left, m_right, T_values, m_values

def run_single_realization(lam, alpha, realization_idx, output_dir):
    """Esegue una singola realizzazione per dati λ e α"""
    
    P = int(alpha * N)
    
    if P < patterns_per_mixed:
        print(f"  ⚠ Warning: P={P} < {patterns_per_mixed}, skipping")
        return None
    
    print(f"\n  → λ={lam:.2f}, α={alpha:.2f}, Real={realization_idx+1}")
    print(f"     P = {P}", end='')
    
    # Genera pattern originali
    eta_original = generate_patterns(P, N, lam, s_e, s_t, T_threshold)
    
    # Crea stati misti
    eta_new, P_new = create_all_mixed_patterns(
        eta_original, patterns_per_mixed, f, sigma_gaussian
    )
    
    if eta_new is None:
        print(f" → Cannot create mixed patterns, skipping")
        return None
    
    print(f" → P_new = {P_new}", end='')
    
    # Costruisci J_new
    J_new = build_J_matrix(eta_new, f, N)
    
    # Target
    target_mixed_idx = P_new // 2
    target_mixed_pattern = eta_new[target_mixed_idx].copy()
    
    # TROVA TRANSIZIONE DI FASE
    T_c, m_high, m_low, T_tested, m_tested = find_phase_transition(
        J_new, target_mixed_pattern, f, theta,
        T_min=T_initial_min, T_max=T_initial_max,
        max_iterations=15, n_sweeps=n_sweep_mc
    )
    
    if T_c is None:
        print(f" → No clear transition (m_low={m_high:.3f}, m_high={m_low:.3f})")
        T_c = np.nan
    else:
        print(f" → T_c={T_c:.4f}, m_high={m_high:.3f}, m_low={m_low:.3f}")
    
    # Storage risultati
    results = {
        'P_original': P,
        'P_new': P_new,
        'target_idx': target_mixed_idx,
        'T_c': T_c,
        'm_high': m_high,
        'm_low': m_low,
        'T_tested': T_tested,
        'm_tested': m_tested,
        'n_evaluations': len(T_tested)
    }
    
    # Salva file con curve complete
    base_name = f"lambda{lam:.2f}_alpha{alpha:.2f}_real{realization_idx}"
    
    output_filename = os.path.join(output_dir, f"{base_name}.txt")
    with open(output_filename, 'w') as outfile:
        outfile.write("=" * 100 + "\n")
        outfile.write(f"HIERARCHICAL PHASE TRANSITION (λ={lam:.2f}, α={alpha:.2f}, Real={realization_idx})\n")
        outfile.write("=" * 100 + "\n")
        outfile.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        outfile.write(f"P_original = {P}, P_new = {P_new}\n")
        outfile.write(f"Target mixed pattern: {target_mixed_idx}\n\n")
        outfile.write(f"Phase transition results:\n")
        outfile.write(f"  T_c = {T_c if not np.isnan(T_c) else 'Not found'}\n")
        outfile.write(f"  m_high (T→0) = {m_high:.6f}\n")
        outfile.write(f"  m_low (T→∞) = {m_low:.6f}\n")
        outfile.write(f"  Evaluations: {len(T_tested)}\n\n")
        outfile.write("=" * 100 + "\n\n")
        outfile.write(f"{'T':>12} {'m_target':>12}\n")
        outfile.write("-" * 100 + "\n")
        
        # Ordina per temperatura
        sorted_idx = np.argsort(T_tested)
        for idx in sorted_idx:
            outfile.write(f"{T_tested[idx]:12.6f} {m_tested[idx]:12.6f}\n")
    
    # Plot individuale
    if len(T_tested) > 2:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        sorted_idx = np.argsort(T_tested)
        T_sorted = [T_tested[i] for i in sorted_idx]
        m_sorted = [m_tested[i] for i in sorted_idx]
        
        ax.plot(T_sorted, m_sorted, 'o-', linewidth=2.5, markersize=8, 
               color='darkblue', alpha=0.8)
        
        if not np.isnan(T_c):
            ax.axvline(x=T_c, color='red', linestyle='--', linewidth=2.5, 
                      alpha=0.7, label=f'T_c = {T_c:.4f}')
            ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        ax.fill_between(T_sorted, 0, 0.1, color='red', alpha=0.1)
        
        ax.set_xlabel('Temperature', fontsize=14, fontweight='bold')
        ax.set_ylabel('m_target', fontsize=14, fontweight='bold')
        ax.set_title(f'Phase Transition | λ={lam:.2f}, α={alpha:.2f}, P={P}→{P_new}\n{len(T_tested)} evaluations',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}.png"), 
                   dpi=120, bbox_inches='tight')
        plt.close()
    
    return results

# ==================== MAIN LOOP ====================
print("\nInizio scan parametri...\n")

total_runs = len(lambda_values) * len(alpha_values) * n_realizations
current_run = 0

all_results = []

for lam in lambda_values:
    for alpha in alpha_values:
        for real_idx in range(n_realizations):
            current_run += 1
            print(f"[{current_run}/{total_runs}]", end=' ')
            
            result = run_single_realization(lam, alpha, real_idx, output_dir)
            
            if result is not None:
                all_results.append({
                    'lambda': lam,
                    'alpha': alpha,
                    'realization': real_idx,
                    'results': result
                })

print("\n" + "=" * 80)
print("GENERAZIONE PLOT AGGREGATI")
print("=" * 80)

# ==================== PLOT 1: Heatmap T_c vs λ,α ====================
print("Generazione plot 1: Heatmap T_c...")

lambda_grid = np.array(lambda_values)
alpha_grid = np.array(alpha_values)
T_c_grid = np.zeros((len(alpha_values), len(lambda_values)))
T_c_std = np.zeros((len(alpha_values), len(lambda_values)))

for i, alpha in enumerate(alpha_values):
    for j, lam in enumerate(lambda_values):
        T_vals = []
        for res in all_results:
            if abs(res['lambda'] - lam) < 0.01 and abs(res['alpha'] - alpha) < 0.01:
                if not np.isnan(res['results']['T_c']):
                    T_vals.append(res['results']['T_c'])
        
        if T_vals:
            T_c_grid[i, j] = np.mean(T_vals)
            T_c_std[i, j] = np.std(T_vals)
        else:
            T_c_grid[i, j] = np.nan

fig, ax = plt.subplots(figsize=(14, 10))

im = ax.imshow(T_c_grid, aspect='auto', origin='lower',
               cmap='hot', vmin=0, vmax=0.3,
               extent=[lambda_grid[0], lambda_grid[-1], 
                      alpha_grid[0], alpha_grid[-1]],
               interpolation='bilinear')

# Contorni
mask = ~np.isnan(T_c_grid)
if np.any(mask):
    contours = ax.contour(lambda_grid, alpha_grid, T_c_grid,
                          levels=6, colors='cyan', linewidths=2, alpha=0.8)
    ax.clabel(contours, inline=True, fontsize=10, fmt='%.3f')

ax.set_xlabel('λ (Spatial Correlation)', fontsize=15, fontweight='bold')
ax.set_ylabel('α (Load)', fontsize=15, fontweight='bold')
ax.set_title(f'Critical Temperature T_c vs λ and α\n(Hierarchical retrieval, averaged over {n_realizations} realizations)',
             fontsize=16, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax, label='T_c')
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Critical Temperature T_c', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_Tc.png'), 
           dpi=150, bbox_inches='tight')
plt.close()
print("✓ Salvato: heatmap_Tc.png")

# ==================== PLOT 2: Heatmap m_high vs λ,α ====================
print("Generazione plot 2: Heatmap m_high...")

m_high_grid = np.zeros((len(alpha_values), len(lambda_values)))

for i, alpha in enumerate(alpha_values):
    for j, lam in enumerate(lambda_values):
        m_vals = []
        for res in all_results:
            if abs(res['lambda'] - lam) < 0.01 and abs(res['alpha'] - alpha) < 0.01:
                m_vals.append(res['results']['m_high'])
        
        if m_vals:
            m_high_grid[i, j] = np.mean(m_vals)
        else:
            m_high_grid[i, j] = np.nan

fig, ax = plt.subplots(figsize=(14, 10))

im = ax.imshow(m_high_grid, aspect='auto', origin='lower',
               cmap='RdYlBu_r', vmin=0, vmax=1,
               extent=[lambda_grid[0], lambda_grid[-1], 
                      alpha_grid[0], alpha_grid[-1]],
               interpolation='bilinear')

contours = ax.contour(lambda_grid, alpha_grid, m_high_grid,
                      levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                      colors='black', linewidths=2, alpha=0.6)
ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')

ax.set_xlabel('λ (Spatial Correlation)', fontsize=15, fontweight='bold')
ax.set_ylabel('α (Load)', fontsize=15, fontweight='bold')
ax.set_title(f'Retrieval Quality m_high (T→0) vs λ and α\n(Hierarchical retrieval)',
             fontsize=16, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax, label='m_high')
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Overlap at T→0', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_mhigh.png'), 
           dpi=150, bbox_inches='tight')
plt.close()
print("✓ Salvato: heatmap_mhigh.png")

# ==================== PLOT 3: T_c e n_evaluations vs λ ====================
print("Generazione plot 3: T_c vs λ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

for alpha in alpha_values[::2]:
    T_c_vals = []
    n_eval_vals = []
    lambdas_plot = []
    
    for lam in lambda_values:
        T_vals = []
        n_vals = []
        
        for res in all_results:
            if abs(res['lambda'] - lam) < 0.01 and abs(res['alpha'] - alpha) < 0.01:
                if not np.isnan(res['results']['T_c']):
                    T_vals.append(res['results']['T_c'])
                n_vals.append(res['results']['n_evaluations'])
        
        if T_vals:
            lambdas_plot.append(lam)
            T_c_vals.append(np.mean(T_vals))
            n_eval_vals.append(np.mean(n_vals))
    
    if lambdas_plot:
        ax1.plot(lambdas_plot, T_c_vals, 'o-', linewidth=2.5, markersize=7,
                label=f'α={alpha:.2f}', alpha=0.8)
        ax2.plot(lambdas_plot, n_eval_vals, 's-', linewidth=2.5, markersize=7,
                label=f'α={alpha:.2f}', alpha=0.8)

ax1.set_xlabel('λ', fontsize=13, fontweight='bold')
ax1.set_ylabel('T_c', fontsize=13, fontweight='bold')
ax1.set_title('Critical Temperature', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('λ', fontsize=13, fontweight='bold')
ax2.set_ylabel('Number of Evaluations', fontsize=13, fontweight='bold')
ax2.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='best')
ax2.grid(True, alpha=0.3)

plt.suptitle('Phase Transition Analysis',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Tc_and_efficiency.png'),
           dpi=150, bbox_inches='tight')
plt.close()
print("✓ Salvato: Tc_and_efficiency.png")

# ==================== SUMMARY ====================
print("\nGenerazione file summary...")

summary_filename = os.path.join(output_dir, 'summary.txt')
with open(summary_filename, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("HIERARCHICAL PHASE TRANSITION SCAN SUMMARY\n")
    f.write("=" * 100 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Method: ADAPTIVE bisection search for phase transition\n")
    f.write(f"  λ range: {lambda_values[0]:.2f} - {lambda_values[-1]:.2f} ({len(lambda_values)} points)\n")
    f.write(f"  α range: {alpha_values[0]:.2f} - {alpha_values[-1]:.2f} ({len(alpha_values)} points)\n")
    f.write(f"  Realizations: {n_realizations}\n")
    f.write(f"  Total runs: {total_runs}\n")
    f.write(f"  Successful: {len(all_results)}\n\n")
    
    total_evals = sum([res['results']['n_evaluations'] for res in all_results])
    avg_evals = total_evals / len(all_results) if all_results else 0
    f.write(f"Efficiency:\n")
    f.write(f"  Total evaluations: {total_evals}\n")
    f.write(f"  Average per run: {avg_evals:.1f}\n")
    f.write(f"  Speedup vs full scan (50 temps): ~{50/avg_evals:.1f}x\n\n")
    
    f.write("=" * 100 + "\n\n")
    f.write(f"{'λ':>8} {'α':>8} {'Real':>6} {'P_orig':>8} {'P_new':>8} {'T_c':>10} {'m_high':>10} {'n_eval':>8}\n")
    f.write("-" * 100 + "\n")
    
    for res in all_results:
        lam = res['lambda']
        alpha = res['alpha']
        real = res['realization']
        P_orig = res['results']['P_original']
        P_new = res['results']['P_new']
        T_c = res['results']['T_c']
        m_high = res['results']['m_high']
        n_eval = res['results']['n_evaluations']
        
        T_c_str = f"{T_c:.4f}" if not np.isnan(T_c) else "N/A"
        f.write(f"{lam:8.2f} {alpha:8.2f} {real:6d} {P_orig:8d} {P_new:8d} {T_c_str:>10} {m_high:10.4f} {n_eval:8d}\n")

print(f"✓ Salvato: summary.txt")

print("\n" + "=" * 80)
print("SCAN COMPLETATO!")
print("=" * 80)
print(f"Directory output: {output_dir}/")
print(f"Runs completati: {len(all_results)}/{total_runs}")
print(f"Speedup: ~{50/avg_evals:.1f}x rispetto a full temperature scan")
print(f"\nFile generati:")
print(f"  - {len(all_results)} file .txt")
print(f"  - {len(all_results)} plot individuali")
print(f"  - 3 plot aggregati")
print(f"  - 1 summary file")
print("=" * 80)
