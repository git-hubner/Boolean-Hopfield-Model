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
target_pattern = 100
theta = 0.6

# RANGE DI PARAMETRI DA ESPLORARE
lambda_values = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
alpha_values = [0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
n_realizations = 3  # Numero di realizzazioni per ogni combinazione

# Range temperature
T_range = np.linspace(0.001, 0.5, 50)

# Range pattern da visualizzare
pattern_range = range(50, 151)

# Crea cartella output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"mixed_pattern_scan_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print(f"SCAN COMPLETO: λ × α × Realizzazioni")
print("=" * 80)
print(f"λ values: {lambda_values}")
print(f"α values: {alpha_values}")
print(f"Realizzazioni per combinazione: {n_realizations}")
print(f"Temperature: {len(T_range)} punti da {T_range[0]:.3f} a {T_range[-1]:.3f}")
print(f"Output directory: {output_dir}/")
print("=" * 80)

def prob(h, tht, beta):
    with np.errstate(over='ignore'):
        return 1/(1 + np.exp(-(h - tht) * beta))

def magn(Z, eta_p, f):
    return np.dot(eta_p - f, Z) / ((1 - f) * np.sum(eta_p))

def generate_patterns(P, N, lam, s_e, s_t, T_threshold):
    eta = np.zeros((P, N))
    I_ep = np.random.normal(0, s_e, size=(P, N))
    I_t = np.zeros((P, N))
    
    I_t[0] = s_t * np.sqrt(1 - lam**2) * np.random.normal(size=N)
    for i in range(1, P):
        z_t = np.random.normal(size=N)
        I_t[i] = lam * I_t[i-1] + s_t * np.sqrt(1 - lam**2) * z_t
    
    eta = (I_t + I_ep - T_threshold >= 0).astype(float)
    return eta

def build_J_matrix(eta, f, N):
    eta_centered = eta - f
    J = np.dot(eta_centered.T, eta_centered) / (f * (1 - f) * N)
    np.fill_diagonal(J, 0)
    return J

def make_custom_weighted_mixed_pattern(eta, start=90, end=110, 
                                       central_patterns=[99, 100, 101],
                                       f=0.1, method='topk'):
    """
    Costruisce un mixed pattern con pesi personalizzati:
    - Pattern centrali (99,100,101): peso 1.0
    - Pattern vicini (10 pattern attorno ai centrali): peso 0.5
    - Pattern rimanenti: peso 0.25
    """
    P, N = eta.shape
    pattern_indices = list(range(start, end + 1))
    L = len(pattern_indices)
    
    # Inizializza pesi
    w = np.zeros(L)
    
    for i, pattern_idx in enumerate(pattern_indices):
        if pattern_idx in central_patterns:
            w[i] = 1.0
        else:
            min_dist = min(abs(pattern_idx - cp) for cp in central_patterns)
            if min_dist <= 5:
                w[i] = 0.5
            else:
                w[i] = 0.25
    
    # Normalizza i pesi
    w = w / w.sum()
    
    # Somma pesata per neurone
    S = np.zeros(N, dtype=float)
    for k, mu in enumerate(pattern_indices):
        S += w[k] * eta[mu]
    
    # Produce binary mixed pattern con activity f
    if method == 'topk':
        k = int(round(f * N))
        idx = np.argsort(S)
        sel = idx[-k:]
        M_bin = np.zeros(N, dtype=float)
        M_bin[sel] = 1.0
    else:
        thr = np.quantile(S, 1.0 - f)
        M_bin = (S > thr).astype(float)
        current = M_bin.sum()
        target_k = int(round(f * N))
        if current != target_k:
            idx = np.argsort(S)
            sel = idx[-target_k:]
            Mb = np.zeros(N, dtype=float)
            Mb[sel] = 1.0
            M_bin = Mb
    
    return M_bin, S, w, pattern_indices

def run_single_realization(lam, alpha, realization_idx, output_dir):
    """Esegue una singola realizzazione per dati λ e α"""
    
    P = int(alpha * N)
    
    # Verifica che ci siano abbastanza pattern
    if P < 111:
        print(f"  ⚠ Warning: P={P} < 111, skipping λ={lam}, α={alpha}, real={realization_idx}")
        return None
    
    print(f"\n  → λ={lam}, α={alpha}, Realization {realization_idx+1}/{n_realizations}")
    print(f"     P = {P}")
    
    # Genera pattern e matrice J
    eta = generate_patterns(P, N, lam, s_e, s_t, T_threshold)
    J = build_J_matrix(eta, f, N)
    
    # Costruisci stato misto
    M, S, w, idxs = make_custom_weighted_mixed_pattern(
        eta, start=90, end=110,
        central_patterns=[99, 100, 101],
        f=0.1, method='topk'
    )
    
    # Storage risultati
    results = {
        'T': [],
        'm_mix': [],
        'm_99': [],
        'm_100': [],
        'm_101': [],
        'q_0': [],
        'spatial_profiles': []
    }
    
    # Filename base
    base_name = f"lambda{lam:.2f}_alpha{alpha:.3f}_real{realization_idx}"
    
    # Inizializza file output
    output_filename = os.path.join(output_dir, f"{base_name}.txt")
    with open(output_filename, 'w') as outfile:
        outfile.write("=" * 100 + "\n")
        outfile.write(f"MIXED PATTERN EVOLUTION (λ={lam}, α={alpha}, Real={realization_idx})\n")
        outfile.write("=" * 100 + "\n")
        outfile.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        outfile.write(f"N = {N}, f = {f}, P = {P}\n")
        outfile.write(f"Lambda = {lam}, Alpha = {alpha}, theta = {theta}\n")
        outfile.write(f"Mixed pattern: 21 patterns [90-110] with custom weights\n\n")
        outfile.write("=" * 100 + "\n\n")
        outfile.write(f"{'T':>10} {'m_mix':>12} {'m_99':>12} {'m_100':>12} {'m_101':>12} {'q_0':>12}\n")
        outfile.write("-" * 100 + "\n")
    
    # Loop su temperature
    for idx, t in enumerate(T_range):
        # Reset stato: stato misto + rumore
        W = M.copy()
        flip_idx = np.random.choice(N, size=N//30, replace=False)
        W[flip_idx] = 1 - W[flip_idx]
        
        # Dinamica Monte Carlo
        for _ in range(30):
            for i in range(N):
                h = np.dot(J[i], W)
                phi = prob(h, theta, 1/t)
                if np.random.rand() < phi:
                    W[i] = 1
                else:
                    W[i] = 0
        
        # Calcola magnetizzazioni
        m_mix = magn(W, M, f)
        m_99 = magn(W, eta[99], f)
        m_100 = magn(W, eta[100], f)
        m_101 = magn(W, eta[101], f)
        q_0 = np.sum(W) / N
        
        # Calcola profilo spaziale completo
        spatial_profile = [magn(W, eta[k], f) for k in pattern_range]
        
        # Salva risultati
        results['T'].append(t)
        results['m_mix'].append(m_mix)
        results['m_99'].append(m_99)
        results['m_100'].append(m_100)
        results['m_101'].append(m_101)
        results['q_0'].append(q_0)
        results['spatial_profiles'].append(spatial_profile)
        
        # Scrivi nel file
        with open(output_filename, 'a') as outfile:
            outfile.write(f"{t:10.6f} {m_mix:12.6f} {m_99:12.6f} {m_100:12.6f} {m_101:12.6f} {q_0:12.6f}\n")
    
    # Converti in array numpy
    T_array = np.array(results['T'])
    spatial_array = np.array(results['spatial_profiles'])
    
    # ==================== GENERA PLOT ====================
    
    # PLOT 1: Evoluzione magnetizzazioni
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    ax1.plot(results['T'], results['m_mix'], 'D-', label='m_mix (stato misto)', 
             linewidth=3.5, markersize=7, color='darkblue', alpha=0.9)
    ax1.plot(results['T'], results['m_99'], 'o-', label='m₉₉', 
             linewidth=2, markersize=5, alpha=0.7, color='orange')
    ax1.plot(results['T'], results['m_100'], 's-', label='m₁₀₀', 
             linewidth=2.5, markersize=6, color='red')
    ax1.plot(results['T'], results['m_101'], '^-', label='m₁₀₁', 
             linewidth=2, markersize=5, alpha=0.7, color='orange')
    
    ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Threshold')
    ax1.fill_between(results['T'], 0, 0.1, color='red', alpha=0.1)
    
    ax1.set_ylabel('Magnetization', fontsize=14, fontweight='bold')
    ax1.set_title(f'Mixed Pattern Evolution | λ={lam}, α={alpha}, Real={realization_idx}',
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(T_array.min(), T_array.max())
    
    ax2.plot(results['T'], results['q_0'], 'o-', linewidth=2.5, 
             markersize=5, color='purple', alpha=0.8)
    ax2.axhline(y=f, color='red', linestyle='--', alpha=0.5, 
               linewidth=2, label=f'f = {f}')
    
    ax2.set_xlabel('Temperature', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Activity (q₀)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(T_array.min(), T_array.max())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_evolution.png"), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # PLOT 2: Heatmap 2D
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(spatial_array.T, aspect='auto', origin='lower', 
                   cmap='RdYlBu_r', vmin=0, vmax=1,
                   extent=[T_array.min(), T_array.max(),
                          pattern_range[0], pattern_range[-1]],
                   interpolation='bilinear')
    
    ax.axhspan(90, 110, alpha=0.2, edgecolor='yellow', 
              facecolor='none', linewidth=3, linestyle='--', 
              label='Mixed pattern range')
    ax.axhline(y=100, color='white', linestyle='-', linewidth=2, alpha=0.8)
    
    contours = ax.contour(T_array, list(pattern_range), spatial_array.T,
                          levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                          colors='black', linewidths=1.5, alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=9, fmt='%.1f')
    
    ax.set_xlabel('Temperature', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pattern Index', fontsize=14, fontweight='bold')
    ax.set_title(f'Magnetization Heatmap | λ={lam}, α={alpha}, Real={realization_idx}',
                 fontsize=16, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im, ax=ax, label='Magnetization')
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Magnetization', fontsize=13, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_heatmap.png"), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"     ✓ Completato (m_mix finale: {results['m_mix'][-1]:.3f})")
    
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
            print(f"[{current_run}/{total_runs}] λ={lam}, α={alpha}, Real={real_idx+1}")
            
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

# ==================== PLOT AGGREGATO: m_mix vs λ per diversi α ====================
print("Generazione plot aggregato: m_mix vs λ...")

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
axes = axes.flatten()

for idx, alpha in enumerate(alpha_values):
    ax = axes[idx]
    
    # Trova temperatura critica (dove m_mix ~ 0.5)
    T_critical_idx = len(T_range) // 2  # temperatura media come riferimento
    
    for lam in lambda_values:
        # Raccogli tutte le realizzazioni per questa combinazione
        m_mix_values = []
        for res in all_results:
            if res['lambda'] == lam and res['alpha'] == alpha:
                m_mix_values.append(res['results']['m_mix'][T_critical_idx])
        
        if m_mix_values:
            mean_m = np.mean(m_mix_values)
            std_m = np.std(m_mix_values)
            ax.errorbar(lam, mean_m, yerr=std_m, marker='o', markersize=8,
                       capsize=5, capthick=2, linewidth=2, alpha=0.8,
                       label=f'λ={lam}' if idx == 0 else '')
    
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('λ', fontsize=12, fontweight='bold')
    ax.set_ylabel('m_mix', fontsize=12, fontweight='bold')
    ax.set_title(f'α = {alpha}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

# Rimuovi subplot extra
for idx in range(len(alpha_values), len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle(f'm_mix vs λ for different α (T≈{T_range[T_critical_idx]:.3f}, avg over {n_realizations} realizations)',
             fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'aggregate_mmix_vs_lambda.png'), 
           dpi=150, bbox_inches='tight')
plt.close()
print("✓ Salvato: aggregate_mmix_vs_lambda.png")

# ==================== SUMMARY FILE ====================
print("\nGenerazione file summary...")

summary_filename = os.path.join(output_dir, 'summary.txt')
with open(summary_filename, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("MIXED PATTERN RETRIEVAL - PARAMETER SCAN SUMMARY\n")
    f.write("=" * 100 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Parameters scanned:\n")
    f.write(f"  λ values: {lambda_values}\n")
    f.write(f"  α values: {alpha_values}\n")
    f.write(f"  Realizations per combination: {n_realizations}\n")
    f.write(f"  Total runs: {total_runs}\n")
    f.write(f"  Successful runs: {len(all_results)}\n\n")
    f.write(f"System parameters:\n")
    f.write(f"  N = {N}\n")
    f.write(f"  f = {f}\n")
    f.write(f"  θ = {theta}\n")
    f.write(f"  Temperature range: {T_range[0]:.3f} - {T_range[-1]:.3f} ({len(T_range)} points)\n\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"{'λ':>8} {'α':>8} {'Real':>6} {'P':>8} {'m_mix(T_low)':>14} {'m_mix(T_high)':>14}\n")
    f.write("-" * 100 + "\n")
    
    for res in all_results:
        lam = res['lambda']
        alpha = res['alpha']
        real = res['realization']
        P = int(alpha * N)
        m_low = res['results']['m_mix'][0]
        m_high = res['results']['m_mix'][-1]
        f.write(f"{lam:8.2f} {alpha:8.3f} {real:6d} {P:8d} {m_low:14.6f} {m_high:14.6f}\n")

print(f"✓ Salvato: summary.txt")

print("\n" + "=" * 80)
print("SCAN COMPLETATO!")
print("=" * 80)
print(f"Directory output: {output_dir}/")
print(f"Total runs completati: {len(all_results)}/{total_runs}")
print(f"File generati:")
print(f"  - {len(all_results)} file .txt con dati numerici")
print(f"  - {len(all_results)*2} plot individuali (.png)")
print(f"  - 1 plot aggregato")
print(f"  - 1 file summary")
print("=" * 80)
