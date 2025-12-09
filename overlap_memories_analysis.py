import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime

# ==================== PARAMETRI ====================
N = 10000
f = 0.1
s_e = 0.7
s_t = np.sqrt(1 - s_e**2)
T_threshold = norm.ppf(1 - f)
theta = 0.6

# Parametri da esplorare
lambda_values = [0.7, 0.75, 0.8]
alpha_values = [0.05, 0.1, 0.2]

# Pattern da analizzare
target_pattern = 100
neighbor_patterns = [98, 99, 101, 102, 103]  # 5 pattern affianco (2 prima, 2 dopo, target+vicini)

# Range temperature
T_range = np.linspace(0.001, 0.5, 30)
n_mc_steps = 30

# File output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"overlap_analysis_{timestamp}.txt"

def prob(h, tht, beta):
    with np.errstate(over='ignore'):
        return 1/(1 + np.exp(-(h - tht) * beta))

def magn(Z, eta_p, f):
    """Magnetizzazione rispetto al pattern p"""
    return np.dot(eta_p - f, Z) / ((1 - f) * np.sum(eta_p))

def overlap_with_pattern(Z, eta_p):
    """
    Calcola l'overlap tra lo stato Z e il pattern eta_p.
    Overlap = frazione di neuroni che hanno lo stesso valore in Z e eta_p
    """
    return np.sum(Z == eta_p) / len(Z)

def overlap_active_neurons(Z, eta_p):
    """
    Calcola l'overlap considerando SOLO i neuroni attivi.
    = (neuroni attivi sia in Z che in eta_p) / (neuroni attivi in Z OR eta_p)
    Questa è più simile a un coefficiente di Jaccard
    """
    both_active = np.sum((Z == 1) & (eta_p == 1))
    either_active = np.sum((Z == 1) | (eta_p == 1))
    
    if either_active == 0:
        return 0.0
    return both_active / either_active

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

# Inizializza file output
with open(output_filename, 'w') as outfile:
    outfile.write("=" * 120 + "\n")
    outfile.write("OVERLAP ANALYSIS BETWEEN MEMORIES DURING DYNAMICS\n")
    outfile.write("=" * 120 + "\n")
    outfile.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    outfile.write(f"N = {N}, f = {f}, theta = {theta}\n")
    outfile.write(f"Target pattern = {target_pattern}\n")
    outfile.write(f"Neighbor patterns = {neighbor_patterns}\n\n")
    outfile.write(f"Lambda values: {lambda_values}\n")
    outfile.write(f"Alpha values: {alpha_values}\n")
    outfile.write(f"Temperatures: {len(T_range)} points from {T_range[0]:.3f} to {T_range[-1]:.3f}\n\n")
    outfile.write("=" * 120 + "\n\n")

print("=" * 80)
print("ANALISI OVERLAP TRA MEMORIE")
print("=" * 80)
print(f"Target pattern: {target_pattern}")
print(f"Neighbor patterns: {neighbor_patterns}")
print(f"Configurazioni: {len(lambda_values)} λ × {len(alpha_values)} α = {len(lambda_values)*len(alpha_values)}")
print("=" * 80)

# Storage risultati
all_results = {}

# Loop su lambda e alpha
for lam in lambda_values:
    for alpha in alpha_values:
        P = int(alpha * N)
        
        print(f"\n{'='*80}")
        print(f"Processing λ = {lam}, α = {alpha} (P = {P})")
        print(f"{'='*80}")
        
        # Genera pattern e matrice J
        eta = generate_patterns(P, N, lam, s_e, s_t, T_threshold)
        J = build_J_matrix(eta, f, N)
        
        # Inizializza stato con pattern target + noise
        W = eta[target_pattern].copy()
        flip_idx = np.random.choice(N, size=N//30, replace=False)
        W[flip_idx] = 1 - W[flip_idx]
        
        # Storage per questa configurazione
        results = {
            'T': [],
            'magnetizations': {target_pattern: []},
            'overlaps_total': {target_pattern: []},
            'overlaps_active': {target_pattern: []},
            'activity': []
        }
        
        # Aggiungi storage per pattern vicini
        for p in neighbor_patterns:
            results['magnetizations'][p] = []
            results['overlaps_total'][p] = []
            results['overlaps_active'][p] = []
        
        # Scrivi header nel file
        with open(output_filename, 'a') as outfile:
            outfile.write(f"\nLAMBDA = {lam:.3f}, ALPHA = {alpha:.3f}\n")
            outfile.write("-" * 120 + "\n")
            header = f"{'T':>10}"
            for p in [target_pattern] + neighbor_patterns:
                header += f" {'m_'+str(p):>10} {'q_tot_'+str(p):>12} {'q_act_'+str(p):>12}"
            header += f" {'activity':>10}"
            outfile.write(header + "\n")
            outfile.write("-" * 120 + "\n")
        
        # Loop su temperature
        for t_idx, t in enumerate(T_range):
            if (t_idx + 1) % 5 == 0:
                print(f"  [{t_idx+1}/{len(T_range)}] T = {t:.4f}")
            
            # Reset stato per ogni temperatura
            W = eta[target_pattern].copy()
            W[flip_idx] = 1 - W[flip_idx]
            
            # Dinamica Monte Carlo
            for _ in range(n_mc_steps):
                for i in range(N):
                    h = np.dot(J[i], W)
                    phi = prob(h, theta, 1/t)
                    if np.random.rand() < phi:
                        W[i] = 1
                    else:
                        W[i] = 0
            
            # Calcola metriche per target e vicini
            activity = np.sum(W) / N
            results['T'].append(t)
            results['activity'].append(activity)
            
            # Target pattern
            m_target = magn(W, eta[target_pattern], f)
            q_tot_target = overlap_with_pattern(W, eta[target_pattern])
            q_act_target = overlap_active_neurons(W, eta[target_pattern])
            
            results['magnetizations'][target_pattern].append(m_target)
            results['overlaps_total'][target_pattern].append(q_tot_target)
            results['overlaps_active'][target_pattern].append(q_act_target)
            
            # Neighbor patterns
            line_data = f"{t:10.6f} {m_target:10.4f} {q_tot_target:12.4f} {q_act_target:12.4f}"
            
            for p in neighbor_patterns:
                m_p = magn(W, eta[p], f)
                q_tot_p = overlap_with_pattern(W, eta[p])
                q_act_p = overlap_active_neurons(W, eta[p])
                
                results['magnetizations'][p].append(m_p)
                results['overlaps_total'][p].append(q_tot_p)
                results['overlaps_active'][p].append(q_act_p)
                
                line_data += f" {m_p:10.4f} {q_tot_p:12.4f} {q_act_p:12.4f}"
            
            line_data += f" {activity:10.4f}"
            
            # Scrivi nel file
            with open(output_filename, 'a') as outfile:
                outfile.write(line_data + "\n")
        
        # Salva risultati
        key = (lam, alpha)
        all_results[key] = results
        
        with open(output_filename, 'a') as outfile:
            outfile.write("\n")
        
        print(f"  ✓ Completato")

print("\n" + "=" * 80)
print("DINAMICA COMPLETATA - Generazione plot...")
print("=" * 80)

# ==================== PLOT 1: MAGNETIZZAZIONI vs T ====================
for lam in lambda_values:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    for ax_idx, alpha in enumerate(alpha_values):
        ax = axes[ax_idx]
        key = (lam, alpha)
        results = all_results[key]
        
        # Plot magnetizzazioni
        ax.plot(results['T'], results['magnetizations'][target_pattern],
               'o-', linewidth=3, markersize=6, label=f'Pattern {target_pattern} (target)',
               color='red', alpha=0.9)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(neighbor_patterns)))
        for p, color in zip(neighbor_patterns, colors):
            ax.plot(results['T'], results['magnetizations'][p],
                   'o-', linewidth=2, markersize=4, label=f'Pattern {p}',
                   color=color, alpha=0.7)
        
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
        ax.set_ylabel('Magnetization', fontsize=12, fontweight='bold')
        ax.set_title(f'α = {alpha}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    plt.suptitle(f'Magnetizations vs Temperature | λ = {lam}',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'magnetizations_lambda{lam:.2f}_{timestamp}.png',
               dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Salvato: magnetizations_lambda{lam:.2f}_{timestamp}.png")

# ==================== PLOT 2: OVERLAP ATTIVI vs T ====================
for lam in lambda_values:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    for ax_idx, alpha in enumerate(alpha_values):
        ax = axes[ax_idx]
        key = (lam, alpha)
        results = all_results[key]
        
        # Plot overlap neuroni attivi
        ax.plot(results['T'], results['overlaps_active'][target_pattern],
               'o-', linewidth=3, markersize=6, label=f'Pattern {target_pattern} (target)',
               color='red', alpha=0.9)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(neighbor_patterns)))
        for p, color in zip(neighbor_patterns, colors):
            ax.plot(results['T'], results['overlaps_active'][p],
                   'o-', linewidth=2, markersize=4, label=f'Pattern {p}',
                   color=color, alpha=0.7)
        
        ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
        ax.set_ylabel('Overlap (Active Neurons)', fontsize=12, fontweight='bold')
        ax.set_title(f'α = {alpha}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    plt.suptitle(f'Active Neurons Overlap vs Temperature | λ = {lam}',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'overlap_active_lambda{lam:.2f}_{timestamp}.png',
               dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Salvato: overlap_active_lambda{lam:.2f}_{timestamp}.png")

# ==================== PLOT 3: OVERLAP TOTALE vs T ====================
for lam in lambda_values:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    for ax_idx, alpha in enumerate(alpha_values):
        ax = axes[ax_idx]
        key = (lam, alpha)
        results = all_results[key]
        
        # Plot overlap totale
        ax.plot(results['T'], results['overlaps_total'][target_pattern],
               'o-', linewidth=3, markersize=6, label=f'Pattern {target_pattern} (target)',
               color='red', alpha=0.9)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(neighbor_patterns)))
        for p, color in zip(neighbor_patterns, colors):
            ax.plot(results['T'], results['overlaps_total'][p],
                   'o-', linewidth=2, markersize=4, label=f'Pattern {p}',
                   color=color, alpha=0.7)
        
        ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
        ax.set_ylabel('Overlap (Total)', fontsize=12, fontweight='bold')
        ax.set_title(f'α = {alpha}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    plt.suptitle(f'Total Overlap vs Temperature | λ = {lam}',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'overlap_total_lambda{lam:.2f}_{timestamp}.png',
               dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Salvato: overlap_total_lambda{lam:.2f}_{timestamp}.png")

# ==================== PLOT 4: HEATMAP OVERLAP A TEMPERATURA FISSATA ====================
# Seleziona alcune temperature chiave
T_keys = [T_range[0], T_range[len(T_range)//3], T_range[2*len(T_range)//3], T_range[-1]]

for lam in lambda_values:
    fig, axes = plt.subplots(len(T_keys), 3, figsize=(16, 4*len(T_keys)))
    
    for t_idx, T_sel in enumerate(T_keys):
        # Trova indice temperatura più vicina
        T_idx = np.argmin(np.abs(np.array(T_range) - T_sel))
        T_actual = T_range[T_idx]
        
        for alpha_idx, alpha in enumerate(alpha_values):
            ax = axes[t_idx, alpha_idx]
            key = (lam, alpha)
            results = all_results[key]
            
            # Estrai overlaps per questa temperatura
            patterns = [target_pattern] + neighbor_patterns
            overlaps = [results['overlaps_active'][p][T_idx] for p in patterns]
            mags = [results['magnetizations'][p][T_idx] for p in patterns]
            
            # Bar plot
            x_pos = np.arange(len(patterns))
            colors_bar = ['red' if p == target_pattern else 'steelblue' for p in patterns]
            
            bars1 = ax.bar(x_pos - 0.2, overlaps, 0.4, label='Overlap (active)',
                          color=colors_bar, alpha=0.7, edgecolor='black')
            bars2 = ax.bar(x_pos + 0.2, mags, 0.4, label='Magnetization',
                          color='orange', alpha=0.7, edgecolor='black')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(p) for p in patterns], fontsize=10)
            ax.set_xlabel('Pattern Index', fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=11, fontweight='bold')
            
            title = f'T={T_actual:.3f}'
            if t_idx == 0:
                title = f'α={alpha}\n' + title
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            if t_idx == 0 and alpha_idx == 2:
                ax.legend(fontsize=9, loc='upper right')
            
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.05)
    
    plt.suptitle(f'Overlap & Magnetization at Key Temperatures | λ = {lam}',
                fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(f'overlap_heatmap_lambda{lam:.2f}_{timestamp}.png',
               dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Salvato: overlap_heatmap_lambda{lam:.2f}_{timestamp}.png")

print("\n" + "=" * 80)
print("ANALISI COMPLETATA!")
print("=" * 80)
print(f"✓ File dati: {output_filename}")
print(f"✓ Plot generati: {3 * len(lambda_values) + len(lambda_values)} file")
print("=" * 80)
