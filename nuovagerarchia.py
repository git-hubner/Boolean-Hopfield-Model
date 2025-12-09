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

# Parametri simulazione FISSI
lam = 0.75
alpha = 0.05
P = int(alpha * N)
theta = 0.6

# Parametri stati misti
patterns_per_mixed = 20  # Numero di pattern da combinare per ogni stato misto
sigma_gaussian = 10.0    # Sigma per pesi gaussiani

# Range temperature
T_range = np.linspace(0.001, 0.5, 50)

# Range pattern da visualizzare (adattato ai nuovi pattern misti)
pattern_range_new = None  # Verrà impostato dopo la creazione degli stati misti

# Crea cartella output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"hierarchical_mixed_patterns_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print(f"HIERARCHICAL MIXED PATTERNS: λ={lam}, α={alpha}")
print("=" * 80)
print(f"N = {N}, f = {f}, P = {P}")
print(f"Patterns per mixed state: {patterns_per_mixed}")
print(f"Expected number of mixed patterns: {P // patterns_per_mixed}")
print(f"Output directory: {output_dir}/")
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
    """
    Crea un singolo pattern misto da eta[start:end+1] con pesi gaussiani
    """
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
    
    return M_bin, w, pattern_indices

def create_all_mixed_patterns(eta, patterns_per_mixed, f, sigma):
    """
    Crea P/patterns_per_mixed stati misti, ciascuno combinando patterns_per_mixed pattern consecutivi
    
    Returns:
    - eta_new: array (P_new, N) con i nuovi pattern misti
    - mixing_info: lista di dict con info su come ogni pattern misto è stato creato
    """
    P, N = eta.shape
    n_mixed = P // patterns_per_mixed
    
    eta_new = np.zeros((n_mixed, N))
    mixing_info = []
    
    print(f"\nCreazione di {n_mixed} stati misti...")
    
    for i in range(n_mixed):
        start = i * patterns_per_mixed
        end = start + patterns_per_mixed - 1
        center = (start + end) / 2.0
        
        M_bin, w, pattern_indices = make_weighted_mixed_pattern(
            eta, start, end, center=center, sigma=sigma, f=f
        )
        
        eta_new[i] = M_bin
        
        mixing_info.append({
            'mixed_idx': i,
            'start': start,
            'end': end,
            'center': center,
            'pattern_indices': pattern_indices,
            'weights': w,
            'sparsity': M_bin.mean()
        })
        
        if (i + 1) % 5 == 0 or i == 0 or i == n_mixed - 1:
            print(f"  Mixed pattern {i}: patterns [{start}-{end}], sparsity={M_bin.mean():.4f}")
    
    return eta_new, mixing_info

# ==================== STEP 1: GENERA PATTERN ORIGINALI ====================
print("\n" + "=" * 80)
print("STEP 1: Generazione pattern originali")
print("=" * 80)

eta_original = generate_patterns(P, N, lam, s_e, s_t, T_threshold)
J_original = build_J_matrix(eta_original, f, N)

print(f"✓ Generati {P} pattern originali")
print(f"✓ Costruita matrice J_original ({N}×{N})")
print(f"  Sparsità media pattern originali: {eta_original.mean():.4f}")

# ==================== STEP 2: CREA STATI MISTI ====================
print("\n" + "=" * 80)
print("STEP 2: Creazione stati misti da pattern originali")
print("=" * 80)

eta_new, mixing_info = create_all_mixed_patterns(
    eta_original, patterns_per_mixed, f, sigma_gaussian
)

P_new = eta_new.shape[0]
print(f"\n✓ Creati {P_new} nuovi pattern misti")
print(f"  Sparsità media stati misti: {eta_new.mean():.4f}")

# ==================== STEP 3: COSTRUISCI NUOVA MATRICE J ====================
print("\n" + "=" * 80)
print("STEP 3: Costruzione nuova matrice J dai pattern misti")
print("=" * 80)

J_new = build_J_matrix(eta_new, f, N)

print(f"✓ Costruita matrice J_new ({N}×{N}) che memorizza {P_new} pattern misti")
print(f"  Formula: J_ij = 1/(Nf(1-f)) * Σ_μ (η_new^μ_i - f)(η_new^μ_j - f)")

# Calcola overlap tra stati misti
overlaps_between_mixed = np.zeros((P_new, P_new))
for i in range(P_new):
    for j in range(P_new):
        overlaps_between_mixed[i, j] = magn(eta_new[i], eta_new[j], f)

print(f"\nStatistiche overlap tra stati misti:")
print(f"  Overlap diagonale (self): {np.mean(np.diag(overlaps_between_mixed)):.4f}")
print(f"  Overlap fuori diagonale medio: {np.mean(overlaps_between_mixed[np.triu_indices(P_new, k=1)]):.4f}")
print(f"  Overlap max (non self): {np.max(overlaps_between_mixed - np.eye(P_new)):.4f}")

# ==================== STEP 4: RETRIEVAL DI UNO STATO MISTO ====================
print("\n" + "=" * 80)
print("STEP 4: Retrieval di uno stato misto")
print("=" * 80)

# Scegli uno stato misto target (centrale)
target_mixed_idx = P_new // 2
target_mixed_pattern = eta_new[target_mixed_idx].copy()

print(f"Target: mixed pattern {target_mixed_idx}")
print(f"  Originato da pattern originali [{mixing_info[target_mixed_idx]['start']}-{mixing_info[target_mixed_idx]['end']}]")
print(f"  Sparsità: {target_mixed_pattern.mean():.4f}")

# Range per visualizzazione profili (pattern misti vicini al target)
pattern_range_new = range(max(0, target_mixed_idx - 10), min(P_new, target_mixed_idx + 11))

# Storage risultati
results = {
    'T': [],
    'm_target': [],  # Overlap con target mixed pattern
    'q_0': [],
    'spatial_profiles': [],  # Overlap con tutti i pattern misti
}

# File output
output_filename = os.path.join(output_dir, f"retrieval_mixed_{target_mixed_idx}.txt")
with open(output_filename, 'w') as outfile:
    outfile.write("=" * 100 + "\n")
    outfile.write(f"HIERARCHICAL RETRIEVAL: Mixed Pattern {target_mixed_idx}\n")
    outfile.write("=" * 100 + "\n")
    outfile.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    outfile.write(f"Original patterns: P = {P}, λ = {lam}, α = {alpha}\n")
    outfile.write(f"Mixed patterns: P_new = {P_new}, patterns_per_mixed = {patterns_per_mixed}\n")
    outfile.write(f"Target mixed pattern: {target_mixed_idx} (from original [{mixing_info[target_mixed_idx]['start']}-{mixing_info[target_mixed_idx]['end']}])\n\n")
    outfile.write("=" * 100 + "\n\n")
    outfile.write(f"{'T':>10} {'m_target':>12} {'q_0':>12}\n")
    outfile.write("-" * 100 + "\n")

print("\nEsecuzione dinamica per tutte le temperature...")

# Loop su temperature
for idx, t in enumerate(T_range):
    if (idx + 1) % 5 == 0:
        print(f"  [{idx+1}/{len(T_range)}] T = {t:.4f}")
    
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
    
    # Calcola overlap con target mixed pattern
    m_target = magn(W, target_mixed_pattern, f)
    q_0 = np.sum(W) / N
    
    # Calcola profilo spaziale: overlap con tutti i pattern misti
    spatial_profile = [magn(W, eta_new[k], f) for k in range(P_new)]
    
    # Salva risultati
    results['T'].append(t)
    results['m_target'].append(m_target)
    results['q_0'].append(q_0)
    results['spatial_profiles'].append(spatial_profile)
    
    # Scrivi nel file
    with open(output_filename, 'a') as outfile:
        outfile.write(f"{t:10.6f} {m_target:12.6f} {q_0:12.6f}\n")

print("✓ Dinamica completata\n")

# Converti in array numpy
T_array = np.array(results['T'])
spatial_array = np.array(results['spatial_profiles'])

# ==================== PLOT 1: EVOLUZIONE m_target E ATTIVITÀ ====================
print("Generazione plot 1: Evoluzione retrieval...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# m_target
ax1.plot(results['T'], results['m_target'], 'D-', 
         linewidth=4, markersize=8, color='darkblue', alpha=0.95,
         label=f'Overlap with target mixed pattern {target_mixed_idx}')

ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Threshold')
ax1.fill_between(results['T'], 0, 0.1, color='red', alpha=0.1)

ax1.set_ylabel('Overlap', fontsize=14, fontweight='bold')
ax1.set_title(f'Hierarchical Retrieval | λ={lam}, α={alpha}, P_new={P_new}',
              fontsize=16, fontweight='bold')
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(T_array.min(), T_array.max())

# Activity
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
plt.savefig(os.path.join(output_dir, 'retrieval_evolution.png'), 
           dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Salvato: retrieval_evolution.png\n")

# ==================== PLOT 2: HEATMAP OVERLAP CON TUTTI I MIXED PATTERNS ====================
print("Generazione plot 2: Heatmap overlap con pattern misti...")

fig, ax = plt.subplots(figsize=(16, 10))

im = ax.imshow(spatial_array.T, aspect='auto', origin='lower', 
               cmap='RdYlBu_r', vmin=0, vmax=1,
               extent=[T_array.min(), T_array.max(), 0, P_new],
               interpolation='bilinear')

# Evidenzia target
ax.axhline(y=target_mixed_idx, color='white', linestyle='-', 
          linewidth=3, alpha=0.9, label=f'Target mixed {target_mixed_idx}')

# Contorni
contours = ax.contour(T_array, range(P_new), spatial_array.T,
                      levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                      colors='black', linewidths=1.5, alpha=0.4)
ax.clabel(contours, inline=True, fontsize=9, fmt='%.1f')

ax.set_xlabel('Temperature', fontsize=16, fontweight='bold')
ax.set_ylabel('Mixed Pattern Index', fontsize=16, fontweight='bold')
ax.set_title(f'Overlap Heatmap: State vs Mixed Patterns | P_new={P_new}',
             fontsize=18, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax, label='Overlap')
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Overlap', fontsize=14, fontweight='bold')

ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_mixed_patterns.png'), 
           dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Salvato: heatmap_mixed_patterns.png\n")

# ==================== PLOT 3: PROFILI SPAZIALI A TEMPERATURE CHIAVE ====================
print("Generazione plot 3: Profili spaziali a temperature chiave...")

m_target_array = np.array(results['m_target'])

# Temperature chiave
idx_high = 0
idx_transition_start = np.argmax(m_target_array < 0.7) if any(m_target_array < 0.7) else len(m_target_array)//3
idx_transition_mid = np.argmax(m_target_array < 0.5) if any(m_target_array < 0.5) else len(m_target_array)//2
idx_transition_end = np.argmax(m_target_array < 0.2) if any(m_target_array < 0.2) else 2*len(m_target_array)//3
idx_low = -1

key_indices = [idx_high, idx_transition_start, idx_transition_mid, 
               idx_transition_end, idx_low]
key_labels = ['Low T\n(High Retrieval)', 'Transition Start', 
              'Mid Transition', 'Transition End', 'High T\n(No Retrieval)']

fig, axes = plt.subplots(1, 5, figsize=(24, 5))

for ax, idx, label in zip(axes, key_indices, key_labels):
    T_val = results['T'][idx]
    profile = results['spatial_profiles'][idx]
    
    ax.plot(range(P_new), profile, linewidth=3, color='steelblue', alpha=0.8)
    ax.fill_between(range(P_new), 0, profile, alpha=0.3, color='steelblue')
    
    # Evidenzia target
    ax.scatter([target_mixed_idx], [results['m_target'][idx]],
               c='red', s=250, zorder=5, marker='*',
               edgecolors='black', linewidth=2)
    
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=target_mixed_idx, color='red', linestyle=':', alpha=0.3, linewidth=2)
    
    info = f'T = {T_val:.4f}\nm_target = {results["m_target"][idx]:.3f}'
    ax.text(0.5, 0.95, info, transform=ax.transAxes, 
           fontsize=11, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax.set_xlabel('Mixed Pattern Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overlap', fontsize=12, fontweight='bold')
    ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, P_new)

plt.suptitle(f'Key Temperature Profiles (Hierarchical) | Target={target_mixed_idx}',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'key_profiles_hierarchical.png'), 
           dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Salvato: key_profiles_hierarchical.png\n")

# ==================== PLOT 4: MATRICE OVERLAP TRA STATI MISTI ====================
print("Generazione plot 4: Matrice overlap tra stati misti...")

fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(overlaps_between_mixed, cmap='RdYlBu_r', vmin=0, vmax=1,
               aspect='auto', origin='lower')

# Evidenzia target
ax.axhline(y=target_mixed_idx, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(x=target_mixed_idx, color='yellow', linestyle='--', linewidth=2, alpha=0.7)

ax.set_xlabel('Mixed Pattern Index', fontsize=14, fontweight='bold')
ax.set_ylabel('Mixed Pattern Index', fontsize=14, fontweight='bold')
ax.set_title(f'Overlap Matrix Between {P_new} Mixed Patterns',
             fontsize=16, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, label='Overlap')
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Overlap', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overlap_matrix_mixed.png'), 
           dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Salvato: overlap_matrix_mixed.png\n")

# ==================== SUMMARY ====================
print("=" * 80)
print("ANALISI COMPLETATA!")
print("=" * 80)
print(f"✓ Directory output: {output_dir}/")
print(f"\nStatistiche finali:")
print(f"  Pattern originali: {P}")
print(f"  Pattern misti creati: {P_new}")
print(f"  Target mixed pattern: {target_mixed_idx}")
print(f"  m_target(T_low): {results['m_target'][0]:.4f}")
print(f"  m_target(T_high): {results['m_target'][-1]:.4f}")
print(f"\nFile generati:")
print(f"  - 1 file .txt con dati retrieval")
print(f"  - 4 plot (.png)")
print("=" * 80)
