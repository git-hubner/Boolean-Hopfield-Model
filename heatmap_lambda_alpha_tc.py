import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime

# ==================== OPZIONI DEBUG ====================
DEBUG_MODE = False  # Cambia a True per test rapidi

if DEBUG_MODE:
    print("⚠️  MODALITÀ DEBUG ATTIVA - Pochi punti per test rapido")
    N = 5000
    alpha_values = np.linspace(0.05, 0.5, 5)
    lambda_values = np.linspace(0.0, 0.9, 6)
    T_range = np.linspace(0.001, 0.25, 20)
    n_mc_steps = 50
else:
    N = 10000
    alpha_values = np.linspace(0.001, 0.5, 20)
    lambda_values = np.linspace(0.0, 0.9, 25)
    T_range = np.linspace(0.001, 0.25, 30)
    n_mc_steps = 35

# Parametri fissi
f = 0.1
s_e = 0.7
s_t = np.sqrt(1 - s_e**2)
theta = 0.6
T_threshold = norm.ppf(1 - f)

# File output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"heatmap_Tc_lambda_alpha_{timestamp}.txt"

@jit(nopython=True)
def monte_carlo_dynamics(J, Z, theta, f, beta, n_steps, N):
    """Dinamica Monte Carlo ottimizzata con Numba"""
    for step in range(n_steps):
        for i in range(N):
            h = np.dot(J[i], Z)
            phi = 1.0 / (1.0 + np.exp(-(h - (theta - f)) * beta))
            if np.random.rand() < phi:
                Z[i] = 1
            else:
                Z[i] = 0
    return Z

def generate_patterns(P, N, lam, s_e, s_t, T_threshold):
    """Genera i pattern in modo vettorializzato"""
    eta = np.zeros((P, N))
    I_ep = np.random.normal(0, s_e, size=(P, N))
    I_t = np.zeros((P, N))
    
    I_t[0] = s_t * np.sqrt(1 - lam**2) * np.random.normal(size=N)
    
    if P > 1:
        z_t = np.random.normal(size=(P-1, N))
        for i in range(1, P):
            I_t[i] = lam * I_t[i-1] + s_t * np.sqrt(1 - lam**2) * z_t[i-1]
    
    eta = (I_t + I_ep - T_threshold >= 0).astype(float)
    return eta

def build_J_matrix(eta, f, N):
    """Costruisce la matrice J in modo vettorializzato"""
    eta_centered = eta - f
    J = np.dot(eta_centered.T, eta_centered) / (f * (1 - f) * N)
    np.fill_diagonal(J, 0)
    return J

def find_critical_temperature(J, eta0, theta, f, N, T_range, n_mc_steps):
    """Trova la temperatura critica con controlli"""
    Z = eta0.copy()
    flip_idx = np.random.choice(N, size=max(1, N//30), replace=False)
    Z[flip_idx] = 1 - Z[flip_idx]
    
    m_vals = np.zeros(len(T_range))
    
    for idx, t in enumerate(T_range):
        Z_temp = eta0.copy()
        Z_temp[flip_idx] = 1 - Z_temp[flip_idx]
        Z_temp = monte_carlo_dynamics(J, Z_temp, theta, f, 1.0/t, n_mc_steps, N)
        m_vals[idx] = np.dot(eta0 - f, Z_temp) / ((1 - f) * np.sum(eta0))
    
    # Controllo: m iniziale deve essere > 0.7
    if m_vals[0] <= 0.5:
        return None, "m_initial_too_low"
    
    # Trova prima transizione da retrieval (m > 0.1) a SG (m < 0.1)
    in_retrieval = False
    for idx in range(len(m_vals)):
        if m_vals[idx] > 0.2:
            in_retrieval = True
        elif in_retrieval and m_vals[idx] < 0.2:
            return T_range[idx], "success"
    
    if in_retrieval:
        return None, "no_transition"
    else:
        return None, "never_retrieval"

# Inizializza file output
with open(output_filename, 'w') as outfile:
    outfile.write("=" * 100 + "\n")
    outfile.write("CRITICAL TEMPERATURE HEATMAP: λ vs α\n")
    outfile.write("=" * 100 + "\n")
    outfile.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    outfile.write(f"DEBUG MODE: {DEBUG_MODE}\n")
    outfile.write(f"N = {N}, f = {f}, theta = {theta}\n")
    outfile.write(f"s_e = {s_e}, s_t = {s_t:.4f}\n\n")
    outfile.write(f"Alpha range: {alpha_values[0]:.3f} - {alpha_values[-1]:.3f} ({len(alpha_values)} points)\n")
    outfile.write(f"Lambda range: {lambda_values[0]:.3f} - {lambda_values[-1]:.3f} ({len(lambda_values)} points)\n")
    outfile.write(f"Temperature range: {T_range[0]:.3f} - {T_range[-1]:.3f} ({len(T_range)} points)\n")
    outfile.write(f"MC steps per T: {n_mc_steps}\n")
    outfile.write(f"Total simulations: {len(alpha_values) * len(lambda_values)}\n\n")
    outfile.write("=" * 100 + "\n\n")

print("=" * 80)
print("HEATMAP λ vs α - TEMPERATURA CRITICA")
print("=" * 80)
if DEBUG_MODE:
    print("⚠️  MODALITÀ DEBUG ATTIVA")
print(f"N = {N}")
print(f"Alpha: {len(alpha_values)} punti da {alpha_values[0]:.2f} a {alpha_values[-1]:.2f}")
print(f"Lambda: {len(lambda_values)} punti da {lambda_values[0]:.2f} a {lambda_values[-1]:.2f}")
print(f"Totale simulazioni: {len(alpha_values) * len(lambda_values)}")
print("=" * 80)

# Inizializza matrice risultati
T_c_matrix = np.full((len(lambda_values), len(alpha_values)), np.nan)
status_matrix = np.full((len(lambda_values), len(alpha_values)), "", dtype=object)

total_sims = len(alpha_values) * len(lambda_values)
sim_count = 0
success_count = 0

# Scrivi header tabella nel file
with open(output_filename, 'a') as outfile:
    outfile.write("RESULTS TABLE:\n")
    outfile.write("-" * 100 + "\n")
    outfile.write(f"{'#':>5} {'alpha':>8} {'lambda':>8} {'P':>8} {'T_c':>12} {'status':>20}\n")
    outfile.write("-" * 100 + "\n")

# Loop principale
for i, lam in enumerate(lambda_values):
    for j, alpha in enumerate(alpha_values):
        sim_count += 1
        P = int(alpha * N)
        
        # Progress update
        if sim_count % 10 == 0 or DEBUG_MODE:
            progress = 100 * sim_count / total_sims
            print(f"[{sim_count:4d}/{total_sims}] ({progress:5.1f}%) | λ={lam:.3f}, α={alpha:.3f}, P={P:4d}", end="")
        
        # Genera pattern e matrice J
        eta = generate_patterns(P, N, lam, s_e, s_t, T_threshold)
        J = build_J_matrix(eta, f, N)
        
        # Trova temperatura critica
        if P > 0 and eta.shape[0] > 0:
            T_c, status = find_critical_temperature(J, eta[min(P-1, 100)], theta, f, N, T_range, n_mc_steps)
        else:
            T_c, status = None, "invalid_P"
        
        # Salva risultati
        T_c_matrix[i, j] = T_c if T_c is not None else np.nan
        status_matrix[i, j] = status
        
        if status == "success":
            success_count += 1
            if sim_count % 10 == 0 or DEBUG_MODE:
                print(f" → T_c={T_c:.4f} ✓")
        else:
            if sim_count % 10 == 0 or DEBUG_MODE:
                print(f" → {status}")
        
        # Scrivi nel file
        with open(output_filename, 'a') as outfile:
            T_c_str = f"{T_c:.6f}" if T_c is not None else "None"
            outfile.write(f"{sim_count:5d} {alpha:8.4f} {lam:8.4f} {P:8d} {T_c_str:>12} {status:>20}\n")

print("\n" + "=" * 80)
print(f"Simulazioni completate: {sim_count}/{total_sims}")
print(f"Transizioni trovate: {success_count}/{total_sims} ({100*success_count/total_sims:.1f}%)")
print("=" * 80)

# Scrivi statistiche nel file
with open(output_filename, 'a') as outfile:
    outfile.write("\n" + "=" * 100 + "\n")
    outfile.write("STATISTICS:\n")
    outfile.write("=" * 100 + "\n")
    outfile.write(f"Total simulations: {total_sims}\n")
    outfile.write(f"Successful transitions: {success_count} ({100*success_count/total_sims:.1f}%)\n\n")
    
    # Conta status
    unique_status, counts = np.unique([s for s in status_matrix.flatten() if s], return_counts=True)
    outfile.write("Status breakdown:\n")
    for status, count in zip(unique_status, counts):
        outfile.write(f"  {status:25s}: {count:5d} ({100*count/total_sims:5.1f}%)\n")
    
    # Statistiche T_c
    valid_Tc = T_c_matrix[~np.isnan(T_c_matrix)]
    if len(valid_Tc) > 0:
        outfile.write(f"\nT_c statistics:\n")
        outfile.write(f"  Mean:   {np.mean(valid_Tc):.6f}\n")
        outfile.write(f"  Std:    {np.std(valid_Tc):.6f}\n")
        outfile.write(f"  Min:    {np.min(valid_Tc):.6f}\n")
        outfile.write(f"  Max:    {np.max(valid_Tc):.6f}\n")
        outfile.write(f"  Median: {np.median(valid_Tc):.6f}\n")

# PLOT: HEATMAP λ vs α con T_c
fig, ax = plt.subplots(figsize=(14, 10))

# Plot heatmap
im = ax.imshow(T_c_matrix, aspect='auto', origin='lower', cmap='RdYlBu_r',
               extent=[alpha_values.min(), alpha_values.max(),
                      lambda_values.min(), lambda_values.max()],
               interpolation='nearest')

# Aggiungi contorni se ci sono abbastanza dati
if success_count > 10:
    # Crea versione interpolata per contorni smooth
    from scipy.ndimage import gaussian_filter
    T_c_smooth = T_c_matrix.copy()
    # Interpola NaN
    mask = ~np.isnan(T_c_smooth)
    if np.sum(mask) > 0:
        # Sostituisci NaN con media per contorni
        T_c_smooth[~mask] = np.nanmean(T_c_smooth)
        T_c_smooth = gaussian_filter(T_c_smooth, sigma=0.5)
        
        # Contorni
        contour_levels = np.linspace(np.nanmin(T_c_matrix), np.nanmax(T_c_matrix), 6)
        contours = ax.contour(alpha_values, lambda_values, T_c_smooth, 
                             levels=contour_levels, colors='black', 
                             linewidths=1.5, alpha=0.6)
        ax.clabel(contours, inline=True, fontsize=9, fmt='%.3f')

# Labels e titolo
ax.set_xlabel(r'$\alpha = P/N$', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$\lambda$ (Temporal Correlation)', fontsize=16, fontweight='bold')
title = f'Critical Temperature $T_c$ Heatmap'
if DEBUG_MODE:
    title += ' [DEBUG MODE]'
ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

# Colorbar
cbar = plt.colorbar(im, ax=ax, label=r'$T_c$ (Critical Temperature)', pad=0.02)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(r'$T_c$ (Critical Temperature)', fontsize=14, fontweight='bold')

# Info box
info_text = f'N = {N}\nf = {f}\nθ = {theta}\n'
info_text += f'Success: {success_count}/{total_sims}'
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'heatmap_Tc_lambda_alpha_{timestamp}.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"\n✓ Heatmap salvata: heatmap_Tc_lambda_alpha_{timestamp}.png")
print(f"✓ Dati salvati: {output_filename}")

# PLOT AGGIUNTIVO: Mappa degli status (per debug)
if DEBUG_MODE:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crea matrice numerica per status
    status_map = {"success": 0, "m_initial_too_low": 1, 
                  "no_transition": 2, "never_retrieval": 3}
    status_numeric = np.full_like(T_c_matrix, -1, dtype=float)
    for i in range(len(lambda_values)):
        for j in range(len(alpha_values)):
            if status_matrix[i, j] in status_map:
                status_numeric[i, j] = status_map[status_matrix[i, j]]
    
    im = ax.imshow(status_numeric, aspect='auto', origin='lower', 
                   cmap='Set3', vmin=-0.5, vmax=3.5,
                   extent=[alpha_values.min(), alpha_values.max(),
                          lambda_values.min(), lambda_values.max()])
    
    ax.set_xlabel(r'$\alpha = P/N$', fontsize=14)
    ax.set_ylabel(r'$\lambda$', fontsize=14)
    ax.set_title('Status Map [DEBUG]', fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Success', 'Low m₀', 'No trans.', 'Never ret.'])
    
    plt.tight_layout()
    plt.savefig(f'status_map_{timestamp}.png', dpi=150)
    plt.show()
    print(f"✓ Status map salvata (debug)")
