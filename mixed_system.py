import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime

# ==================== PARAMETRI ====================
N = 10000
f = 0.1
s_e = 0.7
s_t = np.sqrt(1 - s_e**2)
T_threshold = norm.ppf(1 - f)

# Parametri simulazione
alpha = 0.2
P = int(alpha * N)
target_pattern = 100
theta = 0.6
lam = 0.65  # LAMBDA FISSATO

# Range temperature (molte temperature per vedere l'evoluzione)
T_range = np.linspace(0.001, 0.5, 50)

# Range pattern da visualizzare
pattern_range = range(50, 151)

# File output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"spatial_evolution_lambda{lam:.2f}_{timestamp}.txt"

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

print("=" * 80)
print(f"EVOLUZIONE PROFILI SPAZIALI vs TEMPERATURA")
print("=" * 80)
print(f"λ = {lam} (FISSO)")
print(f"α = {alpha}, P = {P}, N = {N}")
print(f"Temperature: {len(T_range)} punti da {T_range[0]:.3f} a {T_range[-1]:.3f}")
print(f"Pattern range: {pattern_range[0]}-{pattern_range[-1]}")
print("=" * 80)

# Genera pattern e matrice J
print("\nGenerazione pattern e matrice J...")
eta = generate_patterns(P, N, lam, s_e, s_t, T_threshold)
J = build_J_matrix(eta, f, N)
print("✓ Completato")

# Inizializza stato
# ==================== COSTRUZIONE DI UNO STATO MISTO ====================
# Parametri dello stato misto


# =============================
# COSTRUZIONE DELLO STATO MISTO
# =============================
import numpy as np

def make_weighted_mixed_pattern(eta, start=90, end=110, center=None, sigma=10.0,
                                f=0.1, method='topk'):
    """
    Costruisce un mixed pattern binario come combinazione lineare dei pattern
    eta[start:end+1], con pesi centrati in `center` (gaussiana) e attività = f.

    Params
    ------
    eta : np.ndarray, shape (P, N)    # pattern storage
    start, end : int                  # indice iniziale e finale (inclusi)
    center : float or None            # centro della gaussiana (index). se None -> (start+end)/2
    sigma : float                     # dev std della gaussiana dei pesi (in units di index)
    f : float                         # activity target (fractions of neurons ON)
    method : 'topk' or 'quantile'     # come ottenere activity f

    Returns
    -------
    M_bin : np.ndarray (N,)           # mixed binary pattern (0/1) con activity ~ f (or exact f for topk)
    S : np.ndarray (N,)               # somma pesata grezza (float)
    w : np.ndarray (L,)               # pesi usati sui pattern (in ordine pattern_indices)
    pattern_indices : list
    """
    P, N = eta.shape
    pattern_indices = list(range(start, end+1))
    L = len(pattern_indices)
    if center is None:
        center = (start + end) / 2.0

    # gaussian weights centered at `center`
    mus = np.array(pattern_indices)
    # distance in pattern-index units
    dist = mus - center
    w_raw = np.exp(-0.5 * (dist / sigma)**2)
    # normalize weights to sum to 1 (optional; keeps S scale bounded)
    w = w_raw / w_raw.sum()

    # weighted sum per neuron
    S = np.zeros(N, dtype=float)
    for k, mu in enumerate(pattern_indices):
        S += w[k] * eta[mu]

    # produce binary mixed pattern with desired activity f
    if method == 'quantile':
        # threshold at the (1-f) quantile: fraction of S > thr ≈ f
        thr = np.quantile(S, 1.0 - f)
        M_bin = (S > thr).astype(float)
        # small tie handling: if due to ties count differs, adjust by topk fallback
        current = M_bin.sum()
        target_k = int(round(f * N))
        if current != target_k:
            # enforce exact using top-k fallback
            idx = np.argsort(S)
            sel = idx[-target_k:]
            Mb = np.zeros(N, dtype=float)
            Mb[sel] = 1.0
            M_bin = Mb
    elif method == 'topk':
        k = int(round(f * N))
        idx = np.argsort(S)         # ascending
        sel = idx[-k:]              # indices of top-k
        M_bin = np.zeros(N, dtype=float)
        M_bin[sel] = 1.0
    else:
        raise ValueError("method must be 'quantile' or 'topk'")

    return M_bin, S, w, pattern_indices

# ---------- Esempio d'uso ----------
# supponendo eta definito (P x N), p/f dato
# usa i pattern 90..110 (21 pattern) con pesi centrati su 100
M, S, w, idxs = make_weighted_mixed_pattern(eta, start=90, end=110,
                                            center=100, sigma=10.0,
                                            f=0.1, method='quantile')

print("activity M:", M.mean())
print("weights sum:", w.sum())
print("weights (centered):", w)





# Storage risultati
results = {
    'T': [],
    'm_99': [],
    'm_100': [],
    'm_101': [],
    'q_0': [],
    'spatial_profiles': []
}

# Inizializza file output
with open(output_filename, 'w') as outfile:
    outfile.write("=" * 100 + "\n")
    outfile.write(f"SPATIAL PROFILE EVOLUTION vs TEMPERATURE (λ = {lam})\n")
    outfile.write("=" * 100 + "\n")
    outfile.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    outfile.write(f"N = {N}, f = {f}, alpha = {alpha}, P = {P}\n")
    outfile.write(f"Lambda = {lam} (FIXED)\n")
    outfile.write(f"Target pattern = {target_pattern}, theta = {theta}\n\n")
    outfile.write("=" * 100 + "\n\n")
    outfile.write(f"{'T':>10} {'m_99':>12} {'m_100':>12} {'m_101':>12} {'q_0':>12}\n")
    outfile.write("-" * 100 + "\n")

print("\nEsecuzione dinamica per tutte le temperature...")
# Loop su temperature
for idx, t in enumerate(T_range):
    if (idx + 1) % 5 == 0:
        print(f"  [{idx+1}/{len(T_range)}] T = {t:.4f}")
    
    W = M.copy()   # dovrebbe essere ≈ 0.1
    # Disturba lo stato misto con alcune flip random
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
    m_99 = magn(W, eta[99], f)
    m_100 = magn(W, eta[100], f)
    m_101 = magn(W, eta[101], f)
    q_0 = np.sum(W) / N
    
    m_mix = magn(W, M, f)
    print(m_mix)

    # Calcola profilo spaziale completo
    spatial_profile = [magn(W, eta[k], f) for k in pattern_range]
    
    # Salva risultati
    
    results['T'].append(t)
    results['m_99'].append(m_99)
    results['m_100'].append(m_100)
    results['m_101'].append(m_101)
    results['q_0'].append(q_0)
    results['spatial_profiles'].append(spatial_profile)
    
    # Scrivi nel file
    with open(output_filename, 'a') as outfile:
        outfile.write(f"{t:10.6f} {m_99:12.6f} {m_100:12.6f} {m_101:12.6f} {q_0:12.6f}\n")

print("✓ Dinamica completata\n")

# Converti in array numpy per plotting
T_array = np.array(results['T'])
spatial_array = np.array(results['spatial_profiles'])  # Shape: (n_temps, n_patterns)

# ==================== PLOT 1: GRIGLIA DI PROFILI SPAZIALI ====================
print("Generazione plot 1: Griglia profili spaziali...")

# Seleziona 16 temperature distribuite uniformemente
n_plots = 16
temp_indices = np.linspace(0, len(T_range)-1, n_plots, dtype=int)

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for idx, t_idx in enumerate(temp_indices):
    ax = axes[idx]
    T_val = results['T'][t_idx]
    profile = results['spatial_profiles'][t_idx]
    m100 = results['m_100'][t_idx]
    
    # Plot profilo
    ax.plot(pattern_range, profile, linewidth=2, color='steelblue', alpha=0.8)
    
    # Evidenzia pattern centrali
    ax.scatter([99, 100, 101], 
               [results['m_99'][t_idx], m100, results['m_101'][t_idx]],
               c=['orange', 'red', 'orange'], s=100, zorder=5,
               edgecolors='black', linewidth=1.5)
    
    # Linee di riferimento
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.axvline(x=100, color='red', linestyle=':', alpha=0.2, linewidth=1.5)
    
    # Annotazione
    ax.text(0.05, 0.95, f'T={T_val:.3f}\nm₁₀₀={m100:.3f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(pattern_range[0], pattern_range[-1])
    ax.grid(True, alpha=0.2)
    
    if idx >= 12:
        ax.set_xlabel('Pattern', fontsize=10)
    if idx % 4 == 0:
        ax.set_ylabel('Magnetization', fontsize=10)

plt.suptitle(f'Spatial Profile Evolution | λ = {lam}, α = {alpha}', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'spatial_grid_lambda{lam:.2f}_{timestamp}.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Salvato: spatial_grid_lambda{:.2f}_{}.png\n".format(lam, timestamp))

# ==================== PLOT 2: HEATMAP 2D (Pattern vs Temperature) ====================
print("Generazione plot 2: Heatmap 2D...")

fig, ax = plt.subplots(figsize=(16, 10))

# Plot heatmap
im = ax.imshow(spatial_array.T, aspect='auto', origin='lower', 
               cmap='RdYlBu_r', vmin=0, vmax=1,
               extent=[T_array.min(), T_array.max(),
                      pattern_range[0], pattern_range[-1]],
               interpolation='bilinear')

# Evidenzia pattern target
ax.axhline(y=100, color='white', linestyle='--', linewidth=2, alpha=0.8, label='Target (100)')
ax.axhline(y=99, color='yellow', linestyle=':', linewidth=1.5, alpha=0.6)
ax.axhline(y=101, color='yellow', linestyle=':', linewidth=1.5, alpha=0.6)

# Contorni
contours = ax.contour(T_array, list(pattern_range), spatial_array.T,
                      levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                      colors='black', linewidths=1.5, alpha=0.4)
ax.clabel(contours, inline=True, fontsize=9, fmt='%.1f')

ax.set_xlabel('Temperature', fontsize=16, fontweight='bold')
ax.set_ylabel('Pattern Index', fontsize=16, fontweight='bold')
ax.set_title(f'Magnetization Heatmap: Pattern vs Temperature | λ = {lam}, α = {alpha}',
             fontsize=18, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax, label='Magnetization')
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Magnetization', fontsize=14, fontweight='bold')

ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig(f'heatmap_2d_lambda{lam:.2f}_{timestamp}.png', dpi=200, bbox_inches='tight')
plt.show()
print("✓ Salvato: heatmap_2d_lambda{:.2f}_{}.png\n".format(lam, timestamp))

# ==================== PLOT 3: EVOLUZIONE m_99, m_100, m_101 vs T ====================
print("Generazione plot 3: Evoluzione magnetizzazioni centrali...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Magnetizzazioni
ax1.plot(results['T'], results['m_99'], 'o-', label='m₉₉', 
         linewidth=2.5, markersize=5, alpha=0.8, color='orange')
ax1.plot(results['T'], results['m_100'], 's-', label='m₁₀₀ (target)', 
         linewidth=3, markersize=6, color='red')
ax1.plot(results['T'], results['m_101'], '^-', label='m₁₀₁', 
         linewidth=2.5, markersize=5, alpha=0.8, color='orange')

ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Threshold')
ax1.fill_between(results['T'], 0, 0.1, color='red', alpha=0.1)

ax1.set_ylabel('Magnetization', fontsize=14, fontweight='bold')
ax1.set_title(f'Central Patterns Evolution | λ = {lam}, α = {alpha}',
              fontsize=16, fontweight='bold')
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(T_array.min(), T_array.max())

# Subplot 2: Attività q_0
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
plt.savefig(f'magnetization_evolution_lambda{lam:.2f}_{timestamp}.png', 
           dpi=150, bbox_inches='tight')
plt.show()
print("✓ Salvato: magnetization_evolution_lambda{:.2f}_{}.png\n".format(lam, timestamp))

# ==================== PLOT 4: PROFILI SPECIFICI (Temperature chiave) ====================
print("Generazione plot 4: Profili a temperature chiave...")

# Identifica temperature chiave basate su m_100
m100_array = np.array(results['m_100'])

# Trova punti chiave
idx_high = 0  # T bassa, alta magnetizzazione
idx_transition_start = np.argmax(m100_array < 0.7) if any(m100_array < 0.7) else len(m100_array)//3
idx_transition_mid = np.argmax(m100_array < 0.5) if any(m100_array < 0.5) else len(m100_array)//2
idx_transition_end = np.argmax(m100_array < 0.2) if any(m100_array < 0.2) else 2*len(m100_array)//3
idx_low = -1  # T alta, bassa magnetizzazione

key_indices = [idx_high, idx_transition_start, idx_transition_mid, 
               idx_transition_end, idx_low]
key_labels = ['Low T\n(High Retrieval)', 'Transition Start', 
              'Mid Transition', 'Transition End', 'High T\n(No Retrieval)']

fig, axes = plt.subplots(1, 5, figsize=(24, 5))

for ax, idx, label in zip(axes, key_indices, key_labels):
    T_val = results['T'][idx]
    profile = results['spatial_profiles'][idx]
    
    ax.plot(pattern_range, profile, linewidth=3, color='steelblue', alpha=0.8)
    ax.fill_between(pattern_range, 0, profile, alpha=0.3, color='steelblue')
    
    # Evidenzia centrali
    ax.scatter([99, 100, 101],
               [results['m_99'][idx], results['m_100'][idx], results['m_101'][idx]],
               c=['orange', 'red', 'orange'], s=200, zorder=5,
               edgecolors='black', linewidth=2)
    
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=100, color='red', linestyle=':', alpha=0.3, linewidth=2)
    
    info = f'T = {T_val:.4f}\nm₁₀₀ = {results["m_100"][idx]:.3f}'
    ax.text(0.5, 0.95, info, transform=ax.transAxes, 
           fontsize=11, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax.set_xlabel('Pattern Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Magnetization', fontsize=12, fontweight='bold')
    ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(pattern_range[0], pattern_range[-1])

plt.suptitle(f'Key Temperature Profiles | λ = {lam}, α = {alpha}',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'key_profiles_lambda{lam:.2f}_{timestamp}.png', 
           dpi=150, bbox_inches='tight')
plt.show()
print("✓ Salvato: key_profiles_lambda{:.2f}_{}.png\n".format(lam, timestamp))

print("=" * 80)
print("ANALISI COMPLETATA!")
print("=" * 80)
print(f"✓ File dati: {output_filename}")
print(f"✓ 4 grafici salvati con timestamp: {timestamp}")
print("=" * 80)
