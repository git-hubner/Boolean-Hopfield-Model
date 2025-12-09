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

# Range temperature
T_range = np.linspace(0.001, 0.5, 50)

# Crea cartella output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"hierarchical_scan_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print(f"HIERARCHICAL SCAN: λ × α × Realizzazioni")
print("=" * 80)
print(f"λ values: {len(lambda_values)} points from {lambda_values[0]:.2f} to {lambda_values[-1]:.2f}")
print(f"α values: {len(alpha_values)} points from {alpha_values[0]:.2f} to {alpha_values[-1]:.2f}")
print(f"Realizzazioni per combinazione: {n_realizations}")
print(f"Patterns per mixed state: {patterns_per_mixed}")
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
        # Caso speciale: pattern indipendenti
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
    
    # Pesi gaussiani
    mus = np.array(pattern_indices)
    dist = mus - center
    w_raw = np.exp(-0.5 * (dist / sigma)**2)
    w = w_raw / w_raw.sum()
    
    # Somma pesata
    S = np.zeros(N, dtype=float)
    for k, mu in enumerate(pattern_indices):
        S += w[k] * eta[mu]
    
    # Top-k per mantenere sparsità f
    k = int(round(f * N))
    idx = np.argsort(S)
    sel = idx[-k:]
    M_bin = np.zeros(N, dtype=float)
    M_bin[sel] = 1.0
    
    return M_bin, w

def create_all_mixed_patterns(eta, patterns_per_mixed, f, sigma):
    """
    Crea P/patterns_per_mixed stati misti, ciascuno combinando patterns_per_mixed pattern consecutivi
    """
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

def run_single_realization(lam, alpha, realization_idx, output_dir):
    """Esegue una singola realizzazione per dati λ e α"""
    
    P = int(alpha * N)
    
    # Verifica che ci siano abbastanza pattern
    if P < patterns_per_mixed:
        print(f"  ⚠ Warning: P={P} < {patterns_per_mixed}, skipping λ={lam:.2f}, α={alpha:.2f}")
        return None
    
    print(f"\n  → λ={lam:.2f}, α={alpha:.2f}, Real={realization_idx+1}/{n_realizations}")
    print(f"     P = {P}")
    
    # Genera pattern originali
    eta_original = generate_patterns(P, N, lam, s_e, s_t, T_threshold)
    
    # Crea stati misti
    eta_new, P_new = create_all_mixed_patterns(
        eta_original, patterns_per_mixed, f, sigma_gaussian
    )
    
    if eta_new is None:
        print(f"  ⚠ Warning: Cannot create mixed patterns, skipping")
        return None
    
    print(f"     P_new = {P_new} mixed patterns created")
    
    # Costruisci nuova matrice J dai pattern misti
    J_new = build_J_matrix(eta_new, f, N)
    
    # Scegli target (pattern centrale)
    target_mixed_idx = P_new // 2
    target_mixed_pattern = eta_new[target_mixed_idx].copy()
    
    # Storage risultati
    results = {
        'T': [],
        'm_target': [],
        'q_0': [],
        'P_original': P,
        'P_new': P_new,
        'target_idx': target_mixed_idx
    }
    
    # Filename base
    base_name = f"lambda{lam:.2f}_alpha{alpha:.2f}_real{realization_idx}"
    
    # File output
    output_filename = os.path.join(output_dir, f"{base_name}.txt")
    with open(output_filename, 'w') as outfile:
        outfile.write("=" * 100 + "\n")
        outfile.write(f"HIERARCHICAL RETRIEVAL (λ={lam:.2f}, α={alpha:.2f}, Real={realization_idx})\n")
        outfile.write("=" * 100 + "\n")
        outfile.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        outfile.write(f"P_original = {P}, P_new = {P_new}, patterns_per_mixed = {patterns_per_mixed}\n")
        outfile.write(f"Target mixed pattern: {target_mixed_idx}\n\n")
        outfile.write("=" * 100 + "\n\n")
        outfile.write(f"{'T':>10} {'m_target':>12} {'q_0':>12}\n")
        outfile.write("-" * 100 + "\n")
    
    # Loop su temperature
    for idx, t in enumerate(T_range):
        # Reset stato: target pattern + rumore
        W = target_mixed_pattern.copy()
        flip_idx = np.random.choice(N, size=N//30, replace=False)
        W[flip_idx] = 1 - W[flip_idx]
        
        # Dinamica Monte Carlo con J_new
        for _ in range(30):
            for i in range(N):
                h = np.dot(J_new[i], W)
                phi = prob(h, theta, 1/t)
                if np.random.rand() < phi:
                    W[i] = 1
                else:
                    W[i] = 0
        
        # Calcola overlap
        m_target = magn(W, target_mixed_pattern, f)
        q_0 = np.sum(W) / N
        
        # Salva risultati
        results['T'].append(t)
        results['m_target'].append(m_target)
        results['q_0'].append(q_0)
        
        # Scrivi nel file
        with open(output_filename, 'a') as outfile:
            outfile.write(f"{t:10.6f} {m_target:12.6f} {q_0:12.6f}\n")
    
    # Genera plot individuale
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    ax1.plot(results['T'], results['m_target'], 'D-', 
             linewidth=3, markersize=6, color='darkblue', alpha=0.9)
    ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax1.fill_between(results['T'], 0, 0.1, color='red', alpha=0.1)
    ax1.set_ylabel('Overlap with Target', fontsize=13, fontweight='bold')
    ax1.set_title(f'Hierarchical Retrieval | λ={lam:.2f}, α={alpha:.2f}, P={P}→{P_new}',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(min(results['T']), max(results['T']))
    
    ax2.plot(results['T'], results['q_0'], 'o-', linewidth=2, 
             markersize=4, color='purple', alpha=0.8)
    ax2.axhline(y=f, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('Temperature', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Activity (q₀)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min(results['T']), max(results['T']))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}.png"), 
               dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"     ✓ m_target: {results['m_target'][0]:.3f}→{results['m_target'][-1]:.3f}")
    
    return results

# ==================== MAIN LOOP ====================
print("\nInizio scan parametri...\n")

total_runs = len(lambda_values) * len(alpha_values) * n_realizations
current_run = 0

# Storage per tutte le realizzazioni
all_results = []

for lam in lambda_values:
    for alpha in alpha_values:
        for real_idx in range(n_realizations):
            current_run += 1
            print(f"[{current_run}/{total_runs}] λ={lam:.2f}, α={alpha:.2f}, Real={real_idx+1}")
            
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

# ==================== PLOT AGGREGATO 1: Heatmap m_target(T_low) vs λ,α ====================
print("Generazione plot aggregato 1: Heatmap m_target a T bassa...")

# Crea griglia per heatmap
lambda_grid = np.array(lambda_values)
alpha_grid = np.array(alpha_values)
m_target_low_grid = np.zeros((len(alpha_values), len(lambda_values)))
m_target_low_std = np.zeros((len(alpha_values), len(lambda_values)))

for i, alpha in enumerate(alpha_values):
    for j, lam in enumerate(lambda_values):
        m_vals = []
        for res in all_results:
            if abs(res['lambda'] - lam) < 0.01 and abs(res['alpha'] - alpha) < 0.01:
                m_vals.append(res['results']['m_target'][0])  # T bassa
        
        if m_vals:
            m_target_low_grid[i, j] = np.mean(m_vals)
            m_target_low_std[i, j] = np.std(m_vals)
        else:
            m_target_low_grid[i, j] = np.nan

fig, ax = plt.subplots(figsize=(14, 10))

im = ax.imshow(m_target_low_grid, aspect='auto', origin='lower',
               cmap='RdYlBu_r', vmin=0, vmax=1,
               extent=[lambda_grid[0], lambda_grid[-1], 
                      alpha_grid[0], alpha_grid[-1]],
               interpolation='bilinear')

# Contorni
contours = ax.contour(lambda_grid, alpha_grid, m_target_low_grid,
                      levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                      colors='black', linewidths=2, alpha=0.6)
ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')

ax.set_xlabel('λ (Spatial Correlation)', fontsize=15, fontweight='bold')
ax.set_ylabel('α (Load)', fontsize=15, fontweight='bold')
ax.set_title(f'Hierarchical Retrieval: m_target at T_low vs λ and α\n(Averaged over {n_realizations} realizations)',
             fontsize=16, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax, label='m_target')
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Overlap with Target (T_low)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_mtarget_low.png'), 
           dpi=150, bbox_inches='tight')
plt.close()
print("✓ Salvato: heatmap_mtarget_low.png")

# ==================== PLOT AGGREGATO 2: Heatmap m_target(T_high) vs λ,α ====================
print("Generazione plot aggregato 2: Heatmap m_target a T alta...")

m_target_high_grid = np.zeros((len(alpha_values), len(lambda_values)))

for i, alpha in enumerate(alpha_values):
    for j, lam in enumerate(lambda_values):
        m_vals = []
        for res in all_results:
            if abs(res['lambda'] - lam) < 0.01 and abs(res['alpha'] - alpha) < 0.01:
                m_vals.append(res['results']['m_target'][-1])  # T alta
        
        if m_vals:
            m_target_high_grid[i, j] = np.mean(m_vals)
        else:
            m_target_high_grid[i, j] = np.nan

fig, ax = plt.subplots(figsize=(14, 10))

im = ax.imshow(m_target_high_grid, aspect='auto', origin='lower',
               cmap='RdYlBu_r', vmin=0, vmax=1,
               extent=[lambda_grid[0], lambda_grid[-1], 
                      alpha_grid[0], alpha_grid[-1]],
               interpolation='bilinear')

contours = ax.contour(lambda_grid, alpha_grid, m_target_high_grid,
                      levels=[0.05, 0.1, 0.15, 0.2],
                      colors='black', linewidths=2, alpha=0.6)
ax.clabel(contours, inline=True, fontsize=10, fmt='%.2f')

ax.set_xlabel('λ (Spatial Correlation)', fontsize=15, fontweight='bold')
ax.set_ylabel('α (Load)', fontsize=15, fontweight='bold')
ax.set_title(f'Hierarchical Retrieval: m_target at T_high vs λ and α\n(Averaged over {n_realizations} realizations)',
             fontsize=16, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax, label='m_target')
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Overlap with Target (T_high)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_mtarget_high.png'), 
           dpi=150, bbox_inches='tight')
plt.close()
print("✓ Salvato: heatmap_mtarget_high.png")

# ==================== PLOT AGGREGATO 3: m_target vs λ per diversi α ====================
print("Generazione plot aggregato 3: m_target vs λ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Temperature critica
T_critical_idx = len(T_range) // 2

for alpha in alpha_values[::2]:  # Mostra solo alcuni α per chiarezza
    m_low = []
    m_mid = []
    lambdas_plot = []
    
    for lam in lambda_values:
        m_low_vals = []
        m_mid_vals = []
        
        for res in all_results:
            if abs(res['lambda'] - lam) < 0.01 and abs(res['alpha'] - alpha) < 0.01:
                m_low_vals.append(res['results']['m_target'][0])
                m_mid_vals.append(res['results']['m_target'][T_critical_idx])
        
        if m_low_vals:
            lambdas_plot.append(lam)
            m_low.append(np.mean(m_low_vals))
            m_mid.append(np.mean(m_mid_vals))
    
    if lambdas_plot:
        ax1.plot(lambdas_plot, m_low, 'o-', linewidth=2.5, markersize=7,
                label=f'α={alpha:.2f}', alpha=0.8)
        ax2.plot(lambdas_plot, m_mid, 'o-', linewidth=2.5, markersize=7,
                label=f'α={alpha:.2f}', alpha=0.8)

ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax1.set_xlabel('λ', fontsize=13, fontweight='bold')
ax1.set_ylabel('m_target', fontsize=13, fontweight='bold')
ax1.set_title('T_low (High Retrieval)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)

ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax2.set_xlabel('λ', fontsize=13, fontweight='bold')
ax2.set_ylabel('m_target', fontsize=13, fontweight='bold')
ax2.set_title(f'T_mid ≈ {T_range[T_critical_idx]:.3f}', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)

plt.suptitle('Hierarchical Retrieval: m_target vs λ for different α',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mtarget_vs_lambda.png'),
           dpi=150, bbox_inches='tight')
plt.close()
print("✓ Salvato: mtarget_vs_lambda.png")

# ==================== SUMMARY FILE ====================
print("\nGenerazione file summary...")

summary_filename = os.path.join(output_dir, 'summary.txt')
with open(summary_filename, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("HIERARCHICAL PATTERN SCAN SUMMARY\n")
    f.write("=" * 100 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Parameters:\n")
    f.write(f"  λ range: {lambda_values[0]:.2f} - {lambda_values[-1]:.2f} ({len(lambda_values)} points)\n")
    f.write(f"  α range: {alpha_values[0]:.2f} - {alpha_values[-1]:.2f} ({len(alpha_values)} points)\n")
    f.write(f"  Realizations: {n_realizations}\n")
    f.write(f"  Patterns per mixed: {patterns_per_mixed}\n")
    f.write(f"  Total runs: {total_runs}\n")
    f.write(f"  Successful runs: {len(all_results)}\n\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"{'λ':>8} {'α':>8} {'Real':>6} {'P_orig':>8} {'P_new':>8} {'m_low':>10} {'m_high':>10}\n")
    f.write("-" * 100 + "\n")
    
    for res in all_results:
        lam = res['lambda']
        alpha = res['alpha']
        real = res['realization']
        P_orig = res['results']['P_original']
        P_new = res['results']['P_new']
        m_low = res['results']['m_target'][0]
        m_high = res['results']['m_target'][-1]
        f.write(f"{lam:8.2f} {alpha:8.2f} {real:6d} {P_orig:8d} {P_new:8d} {m_low:10.4f} {m_high:10.4f}\n")

print(f"✓ Salvato: summary.txt")

print("\n" + "=" * 80)
print("SCAN COMPLETATO!")
print("=" * 80)
print(f"Directory output: {output_dir}/")
print(f"Total runs completati: {len(all_results)}/{total_runs}")
print(f"\nFile generati:")
print(f"  - {len(all_results)} file .txt con dati")
print(f"  - {len(all_results)} plot individuali")
print(f"  - 3 plot aggregati (2 heatmap + 1 line plot)")
print(f"  - 1 file summary")
print("=" * 80)
