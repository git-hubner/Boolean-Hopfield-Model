import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime

# ==================== OPZIONI ====================
DEBUG_MODE = True  # Cambia a True per test rapidi
N_REALIZATIONS = 3 if not DEBUG_MODE else 2  # Numero di realizzazioni per ogni (λ, α)

if DEBUG_MODE:
    print("⚠️  MODALITÀ DEBUG ATTIVA")
    N = 10000
    alpha_values = np.linspace(0.05, 0.05, 1)
    lambda_values = np.linspace(0.75, 0.75, 1)
    T_range = np.linspace(0.001, 0.25, 40)
    n_mc_steps = 40
else:
    N = 10000
    alpha_values = np.linspace(0.01, 0.5, 15)
    lambda_values = np.linspace(0.0, 0.9, 20)
    T_range = np.linspace(0.001, 0.26, 30)
    n_mc_steps = 15

# Parametri fissi
f = 0.1
s_e = 0.7
s_t = np.sqrt(1 - s_e**2)
theta = 0.6
T_threshold = norm.ppf(1 - f)

# Range pattern per profili spaziali
pattern_range = range(50, 151)

# File output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"heatmap_Tc_multi_real_{timestamp}.txt"

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

def find_transitions(J, eta0, theta, f, N, T_range, n_mc_steps):
    """
    Trova le temperature critiche:
    - T_c1: prima transizione (retrieval → spin glass, m: 0.9→0.5 o m>0.1→m<0.1)
    - T_c2: seconda transizione (spin glass → paramagnetic, m: 0.5→0, solo per λ>0.7)
    """
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
    if m_vals[0] <= 0.3:
        return None, None, "m_initial_too_low", m_vals
    
    T_c1 = None
    T_c2 = None
    
    # PRIMA TRANSIZIONE: retrieval → spin glass
    # Cerca dove m scende sotto 0.1 (standard) O dove scende da >0.7 a <0.5
    in_high = False
    for idx in range(len(m_vals)):
        if m_vals[idx] > 0.7:
            in_high = True
        elif in_high and m_vals[idx] < 0.4:
            T_c1 = T_range[idx]
            break
    
    # Se non trova transizione 0.7→0.5, usa criterio standard 0.1
    if T_c1 is None:
        in_retrieval = False
        for idx in range(len(m_vals)):
            if m_vals[idx] > 0.2:
                in_retrieval = True
            elif in_retrieval and m_vals[idx] < 0.2:
                T_c1 = T_range[idx]
                break
    
    # SECONDA TRANSIZIONE: spin glass → paramagnetic (solo se T_c1 trovata)
    # Cerca dove m scende da ~0.5 a ~0 (sotto 0.05)
    if T_c1 is not None:
        # Parti da dopo la prima transizione
        start_idx = np.argmin(np.abs(T_range - T_c1))
        in_sg = False
        for idx in range(start_idx, len(m_vals)):

            if 0.2 < m_vals[idx] < 0.4:  # Fase spin glass
                in_sg = True
            elif in_sg and m_vals[idx] < 0.1:  # Transizione a paramagnetic
                T_c2 = T_range[idx]
                break
    
    # Status
    if T_c1 is None:
        status = "no_first_transition"
    elif T_c2 is None:
        status = "only_first_transition"
    else:
        status = "both_transitions"
    
    return T_c1, T_c2, status, m_vals

# Inizializza file output
with open(output_filename, 'w') as outfile:
    outfile.write("=" * 120 + "\n")
    outfile.write("CRITICAL TEMPERATURE HEATMAP: λ vs α (MULTIPLE REALIZATIONS)\n")
    outfile.write("=" * 120 + "\n")
    outfile.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    outfile.write(f"DEBUG MODE: {DEBUG_MODE}\n")
    outfile.write(f"N_REALIZATIONS: {N_REALIZATIONS}\n")
    outfile.write(f"N = {N}, f = {f}, theta = {theta}\n")
    outfile.write(f"s_e = {s_e}, s_t = {s_t:.4f}\n\n")
    outfile.write(f"Alpha range: {alpha_values[0]:.3f} - {alpha_values[-1]:.3f} ({len(alpha_values)} points)\n")
    outfile.write(f"Lambda range: {lambda_values[0]:.3f} - {lambda_values[-1]:.3f} ({len(lambda_values)} points)\n")
    outfile.write(f"Total simulations: {len(alpha_values) * len(lambda_values) * N_REALIZATIONS}\n\n")
    outfile.write("=" * 120 + "\n\n")

print("=" * 80)
print("HEATMAP λ vs α - MULTIPLE REALIZZAZIONI")
print("=" * 80)
if DEBUG_MODE:
    print("⚠️  MODALITÀ DEBUG ATTIVA")
print(f"N = {N}, Realizzazioni per punto: {N_REALIZATIONS}")
print(f"Alpha: {len(alpha_values)} punti")
print(f"Lambda: {len(lambda_values)} punti")
print(f"Totale simulazioni: {len(alpha_values) * len(lambda_values) * N_REALIZATIONS}")
print("=" * 80)

# Inizializza matrici risultati
T_c1_mean = np.full((len(lambda_values), len(alpha_values)), np.nan)
T_c1_std = np.full((len(lambda_values), len(alpha_values)), np.nan)
T_c2_mean = np.full((len(lambda_values), len(alpha_values)), np.nan)
T_c2_std = np.full((len(lambda_values), len(alpha_values)), np.nan)

# Storage per profili spaziali (λ > 0.6)
spatial_profiles_storage = {}

total_sims = len(alpha_values) * len(lambda_values) * N_REALIZATIONS
sim_count = 0
success_tc1 = 0
success_tc2 = 0

# Scrivi header tabella
with open(output_filename, 'a') as outfile:
    outfile.write("RESULTS:\n")
    outfile.write("-" * 120 + "\n")
    outfile.write(f"{'alpha':>8} {'lambda':>8} {'Real':>5} {'T_c1':>12} {'T_c2':>12} {'status':>25}\n")
    outfile.write("-" * 120 + "\n")

# Loop principale
for i, lam in enumerate(lambda_values):
    for j, alpha in enumerate(alpha_values):
        P = int(alpha * N)
        
        # Storage per realizzazioni di questo punto
        T_c1_list = []
        T_c2_list = []
        
        # Storage profili spaziali se λ > 0.6
        if lam > 0.6:
            key = (lam, alpha)
            spatial_profiles_storage[key] = []
        
        for real in range(N_REALIZATIONS):
            sim_count += 1
            
            if sim_count % 20 == 0 or DEBUG_MODE:
                progress = 100 * sim_count / total_sims
                print(f"[{sim_count:4d}/{total_sims}] ({progress:5.1f}%) | "
                      f"λ={lam:.3f}, α={alpha:.3f}, real={real+1}/{N_REALIZATIONS}", end="")
            
            # Genera pattern e matrice J (nuova realizzazione)
            eta = generate_patterns(P, N, lam, s_e, s_t, T_threshold)
            J = build_J_matrix(eta, f, N)
            
            # Trova transizioni
            if P > 0 and eta.shape[0] > 0:
                target_idx = min(P-1, 100)
                T_c1, T_c2, status, m_curve = find_transitions(
                    J, eta[target_idx], theta, f, N, T_range, n_mc_steps
                )
            else:
                T_c1, T_c2, status, m_curve = None, None, "invalid_P", None
            
            # Salva T_c valide
            if T_c1 is not None:
                T_c1_list.append(T_c1)
                success_tc1 += 1
            if T_c2 is not None:
                T_c2_list.append(T_c2)
                success_tc2 += 1
            
            # Salva profili spaziali se λ > 0.6 e prima realizzazione
            if lam > 0.6 and real == 0 and status != "invalid_P":
                # Trova indici di transizione per plottare
                if T_c1 is not None:
                    idx_tc1 = np.argmin(np.abs(T_range - T_c1))
                    # Calcola profilo spaziale a T_c1
                    W = eta[target_idx].copy()
                    flip_idx = np.random.choice(N, size=N//30, replace=False)
                    W[flip_idx] = 1 - W[flip_idx]
                    W = monte_carlo_dynamics(J, W, theta, f, 1.0/T_c1, n_mc_steps, N)
                    profile = [np.dot(eta[k] - f, W) / ((1 - f) * np.sum(eta[k])) 
                              for k in pattern_range]
                    spatial_profiles_storage[key].append({
                        'T': T_c1,
                        'type': 'first_transition',
                        'profile': profile,
                        'm_target': m_curve[idx_tc1]
                    })
            
            if sim_count % 20 == 0 or DEBUG_MODE:
                if T_c1 is not None:
                    print(f" → T_c1={T_c1:.4f}", end="")
                    if T_c2 is not None:
                        print(f", T_c2={T_c2:.4f} ✓✓")
                    else:
                        print(" ✓")
                else:
                    print(f" → {status}")
            
            # Scrivi nel file
            with open(output_filename, 'a') as outfile:
                T_c1_str = f"{T_c1:.6f}" if T_c1 is not None else "None"
                T_c2_str = f"{T_c2:.6f}" if T_c2 is not None else "None"
                outfile.write(f"{alpha:8.4f} {lam:8.4f} {real+1:5d} "
                            f"{T_c1_str:>12} {T_c2_str:>12} {status:>25}\n")
        
        # Calcola statistiche per questo punto (λ, α)
        if len(T_c1_list) > 0:
            T_c1_mean[i, j] = np.mean(T_c1_list)
            T_c1_std[i, j] = np.std(T_c1_list) if len(T_c1_list) > 1 else 0
        
        if len(T_c2_list) > 0:
            T_c2_mean[i, j] = np.mean(T_c2_list)
            T_c2_std[i, j] = np.std(T_c2_list) if len(T_c2_list) > 1 else 0

print("\n" + "=" * 80)
print(f"Simulazioni completate: {sim_count}/{total_sims}")
print(f"Prima transizione trovata: {success_tc1}/{total_sims} ({100*success_tc1/total_sims:.1f}%)")
print(f"Seconda transizione trovata: {success_tc2}/{total_sims} ({100*success_tc2/total_sims:.1f}%)")
print("=" * 80)

# Scrivi statistiche
with open(output_filename, 'a') as outfile:
    outfile.write("\n" + "=" * 120 + "\n")
    outfile.write("STATISTICS:\n")
    outfile.write("=" * 120 + "\n")
    outfile.write(f"Total simulations: {total_sims}\n")
    outfile.write(f"First transitions found: {success_tc1} ({100*success_tc1/total_sims:.1f}%)\n")
    outfile.write(f"Second transitions found: {success_tc2} ({100*success_tc2/total_sims:.1f}%)\n\n")

# ==================== PLOT 1: HEATMAP T_c1 (Prima Transizione) ====================
fig, ax = plt.subplots(figsize=(14, 10))

im = ax.imshow(T_c1_mean, aspect='auto', origin='lower', cmap='RdYlBu_r',
               extent=[alpha_values.min(), alpha_values.max(),
                      lambda_values.min(), lambda_values.max()],
               interpolation='bilinear')

# Contorni
if np.sum(~np.isnan(T_c1_mean)) > 10:
    from scipy.ndimage import gaussian_filter
    T_c1_smooth = T_c1_mean.copy()
    mask = ~np.isnan(T_c1_smooth)
    if np.sum(mask) > 0:
        T_c1_smooth[~mask] = np.nanmean(T_c1_smooth)
        T_c1_smooth = gaussian_filter(T_c1_smooth, sigma=0.5)
        levels = np.linspace(np.nanmin(T_c1_mean), np.nanmax(T_c1_mean), 6)
        contours = ax.contour(alpha_values, lambda_values, T_c1_smooth,
                             levels=levels, colors='black', linewidths=1.5, alpha=0.6)
        ax.clabel(contours, inline=True, fontsize=9, fmt='%.3f')

ax.set_xlabel(r'$\alpha = P/N$', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$\lambda$', fontsize=16, fontweight='bold')
title = f'First Critical Temperature $T_{{c1}}$ (Retrieval → SG) | {N_REALIZATIONS} realizations'
ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

cbar = plt.colorbar(im, ax=ax, label=r'$T_{c1}$ (mean)')
cbar.ax.tick_params(labelsize=12)

info = f'N={N}\nRealizations={N_REALIZATIONS}\nSuccess={success_tc1}/{total_sims}'
ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'heatmap_Tc1_mean_{timestamp}.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Salvato: heatmap_Tc1_mean_{timestamp}.png")

# ==================== PLOT 2: HEATMAP T_c2 (Seconda Transizione, λ>0.7) ====================
if success_tc2 > 0:
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(T_c2_mean, aspect='auto', origin='lower', cmap='viridis',
                   extent=[alpha_values.min(), alpha_values.max(),
                          lambda_values.min(), lambda_values.max()],
                   interpolation='bilinear')
    
    ax.axhline(y=0.7, color='white', linestyle='--', linewidth=2, alpha=0.8,
               label='λ = 0.7 threshold')
    
    ax.set_xlabel(r'$\alpha = P/N$', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'$\lambda$', fontsize=16, fontweight='bold')
    title = f'Second Critical Temperature $T_{{c2}}$ (SG → Paramagnetic) | {N_REALIZATIONS} realizations'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, label=r'$T_{c2}$ (mean)')
    cbar.ax.tick_params(labelsize=12)
    
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'heatmap_Tc2_mean_{timestamp}.png', dpi=200, bbox_inches='tight')
    plt.show()
    print(f"✓ Salvato: heatmap_Tc2_mean_{timestamp}.png")

# ==================== PLOT 3: PROFILI SPAZIALI (λ > 0.6) ====================
if len(spatial_profiles_storage) > 0:
    print(f"\nGenerazione profili spaziali per λ > 0.6...")
    
    # Filtra λ > 0.6
    lambda_spatial = [lam for lam in lambda_values if lam > 0.6]
    
    for lam in lambda_spatial:
        # Trova alcuni alpha rappresentativi
        alpha_to_plot = [alpha_values[0], alpha_values[len(alpha_values)//2], alpha_values[-1]]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for ax_idx, alpha in enumerate(alpha_to_plot):
            ax = axes[ax_idx]
            key = (lam, alpha)
            
            if key in spatial_profiles_storage and len(spatial_profiles_storage[key]) > 0:
                data = spatial_profiles_storage[key][0]
                profile = data['profile']
                T_val = data['T']
                m_val = data['m_target']
                
                ax.plot(pattern_range, profile, linewidth=2.5, color='steelblue', alpha=0.8)
                ax.fill_between(pattern_range, 0, profile, alpha=0.3, color='steelblue')
                
                # Evidenzia pattern centrali se nel range
                if 99 in pattern_range and 100 in pattern_range and 101 in pattern_range:
                    idx_99 = list(pattern_range).index(99)
                    idx_100 = list(pattern_range).index(100)
                    idx_101 = list(pattern_range).index(101)
                    ax.scatter([99, 100, 101],
                              [profile[idx_99], profile[idx_100], profile[idx_101]],
                              c=['orange', 'red', 'orange'], s=150, zorder=5,
                              edgecolors='black', linewidth=2)
                
                ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2)
                ax.axvline(x=100, color='red', linestyle=':', alpha=0.3, linewidth=2)
                
                info = f'α = {alpha:.3f}\nT_c1 = {T_val:.4f}\nm = {m_val:.3f}'
                ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=10, va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
                
                ax.set_xlabel('Pattern Index', fontsize=12, fontweight='bold')
                ax.set_ylabel('Magnetization', fontsize=12, fontweight='bold')
                ax.set_title(f'α = {alpha:.3f}', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(pattern_range[0], pattern_range[-1])
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=14)
                ax.set_xlim(pattern_range[0], pattern_range[-1])
                ax.set_ylim(0, 1)
        
        plt.suptitle(f'Spatial Profiles at First Transition | λ = {lam:.3f}',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'spatial_profiles_lambda{lam:.3f}_{timestamp}.png',
                   dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  ✓ Salvato: spatial_profiles_lambda{lam:.3f}_{timestamp}.png")

print(f"\n✓ Tutti i file salvati con timestamp: {timestamp}")
print(f"✓ File dati: {output_filename}")
