import numpy as np
from scipy.stats import norm
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from datetime import datetime

# ==================== PARAMETRI ====================
DEBUG_MODE = True  # Cambia a False per simulazione completa

if DEBUG_MODE:
    print("⚠️  MODALITÀ DEBUG ATTIVA")
    N = 3000
    alpha_values = [0.05, 0.1]
    lambda_values = [0.7, 0.8]
    T_range = np.linspace(0.001, 0.5, 20)
    n_eigenvectors = 10  # Numero autovettori da analizzare
else:
    N = 10000
    alpha_values = [0.05, 0.1, 0.2]
    lambda_values = [0.7, 0.75, 0.8]
    T_range = np.linspace(0.001, 0.5, 30)
    n_eigenvectors = 20

# Parametri fissi
f = 0.1  # Target activity
s_e = 0.7
s_t = np.sqrt(1 - s_e**2)
T_threshold = norm.ppf(1 - f)
theta = 0.6
n_mc_steps = 30

# File output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"eigenvector_retrieval_{timestamp}.txt"

def prob(h, tht, beta):
    with np.errstate(over='ignore'):
        return 1/(1 + np.exp(-(h - tht) * beta))

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

def binarize_eigenvector_fixed_activity(v, target_activity=0.1):
    """
    Binarizza l'autovettore imponendo esattamente il 10% di neuroni attivi.
    Prende i neuroni con i valori più alti dell'autovettore.
    """
    N = len(v)
    n_active = int(target_activity * N)
    
    # Trova gli indici dei top n_active valori
    top_indices = np.argsort(v)[-n_active:]
    
    # Crea stato binario
    binary_state = np.zeros(N)
    binary_state[top_indices] = 1
    
    return binary_state

def magnetization(W, pattern, f):
    """
    Calcola la magnetizzazione dello stato W rispetto al pattern.
    m = <(ξ - f) · W> / ((1-f) * Σξ)
    """
    return np.dot(pattern - f, W) / ((1 - f) * np.sum(pattern))

def retrieve_from_state(J, initial_state, T, theta, f, N, n_mc_steps):
    """
    Esegue retrieval partendo da uno stato iniziale generico
    """
    W = initial_state.copy()
    
    for _ in range(n_mc_steps):
        for i in range(N):
            h = np.dot(J[i], W)
            phi = prob(h, theta, 1/T)
            if np.random.rand() < phi:
                W[i] = 1
            else:
                W[i] = 0
    
    return W

def activity(W):
    """Frazione neuroni attivi"""
    return np.sum(W) / len(W)

# Inizializza file output
with open(output_filename, 'w') as outfile:
    outfile.write("=" * 120 + "\n")
    outfile.write("EIGENVECTOR RETRIEVAL ANALYSIS - MAGNETIZATION vs TEMPERATURE\n")
    outfile.write("=" * 120 + "\n")
    outfile.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    outfile.write(f"DEBUG MODE: {DEBUG_MODE}\n")
    outfile.write(f"N = {N}, f = {f}, theta = {theta}\n")
    outfile.write(f"Eigenvector activity: {f} (10% neurons active)\n")
    outfile.write(f"Lambda values: {lambda_values}\n")
    outfile.write(f"Alpha values: {alpha_values}\n")
    outfile.write(f"N eigenvectors analyzed: {n_eigenvectors}\n\n")
    outfile.write("=" * 120 + "\n\n")

print("=" * 80)
print("RETRIEVAL SU AUTOVETTORI DELLA MATRICE J")
print("=" * 80)
if DEBUG_MODE:
    print("⚠️  MODALITÀ DEBUG")
print(f"N = {N}")
print(f"Attività autovettori: {f*100:.0f}% (fissata)")
print(f"Autovettori da analizzare: {n_eigenvectors} (maggiori autovalori)")
print(f"Configurazioni: {len(lambda_values)} λ × {len(alpha_values)} α")
print("=" * 80)

# Storage risultati
all_results = {}

for lam in lambda_values:
    for alpha in alpha_values:
        P = int(alpha * N)
        
        print(f"\n{'='*80}")
        print(f"Processing λ = {lam}, α = {alpha} (P = {P})")
        print(f"{'='*80}")
        
        # Genera pattern e matrice J
        print("  1. Generazione pattern e matrice J...")
        eta = generate_patterns(P, N, lam, s_e, s_t, T_threshold)
        J = build_J_matrix(eta, f, N)
        
        # Diagonalizza J
        print("  2. Diagonalizzazione matrice J...")
        eigenvalues, eigenvectors = eigh(J)
        
        # Ordina per autovalori decrescenti
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print(f"     ✓ Autovalori range: [{eigenvalues[-1]:.4f}, {eigenvalues[0]:.4f}]")
        print(f"     ✓ Top 5 autovalori: {eigenvalues[:5]}")
        
        # Binarizza i primi n_eigenvectors autovettori con attività fissa al 10%
        print(f"  3. Binarizzazione top {n_eigenvectors} autovettori (attività = {f})...")
        eigen_states = []
        for i in range(min(n_eigenvectors, N)):
            v = eigenvectors[:, i]
            # Binarizza con attività fissa al 10%
            v_binary = binarize_eigenvector_fixed_activity(v, target_activity=f)
            eigen_states.append(v_binary)
            
            if i < 5:
                actual_activity = np.sum(v_binary) / N
                print(f"     Autovettore {i}: {int(np.sum(v_binary))}/{N} neuroni attivi "
                      f"({100*actual_activity:.1f}%), λ={eigenvalues[i]:.4f}")
        
        # Storage risultati
        results = {
            'T': [],
            'eigenvalues': eigenvalues[:n_eigenvectors],
            'magnetizations': {i: [] for i in range(n_eigenvectors)},
            'q_0': {i: [] for i in range(n_eigenvectors)}
        }
        
        # Scrivi info nel file
        with open(output_filename, 'a') as outfile:
            outfile.write(f"\nLAMBDA = {lam:.3f}, ALPHA = {alpha:.3f}\n")
            outfile.write("-" * 120 + "\n")
            outfile.write(f"Top {n_eigenvectors} eigenvalues:\n")
            for i in range(n_eigenvectors):
                outfile.write(f"  λ_{i} = {eigenvalues[i]:.6f}\n")
            outfile.write("\n")
            outfile.write(f"{'T':>10}")
            for i in range(min(5, n_eigenvectors)):
                outfile.write(f" {'m_'+str(i):>10} {'q0_'+str(i):>10}")
            outfile.write("\n")
            outfile.write("-" * 120 + "\n")
        
        # Loop su temperature
        print(f"  4. Retrieval su autovettori a diverse temperature...")
        for t_idx, t in enumerate(T_range):
            if (t_idx + 1) % 5 == 0:
                print(f"     [{t_idx+1}/{len(T_range)}] T = {t:.4f}")
            
            results['T'].append(t)
            line_data = f"{t:10.6f}"
            
            # Esegui retrieval per ogni autovettore
            for i in range(n_eigenvectors):
                # Aggiungi noise all'autovettore binarizzato
                initial_state = eigen_states[i].copy()
                n_flip = max(1, N // 30)
                flip_idx = np.random.choice(N, size=n_flip, replace=False)
                initial_state[flip_idx] = 1 - initial_state[flip_idx]
                
                # Retrieval
                W = retrieve_from_state(J, initial_state, t, theta, f, N, n_mc_steps)
                
                # MAGNETIZZAZIONE rispetto all'autovettore originale
                m = magnetization(W, eigen_states[i], f)
                results['magnetizations'][i].append(m)
                
                # ATTIVITÀ q_0
                q0 = activity(W)
                results['q_0'][i].append(q0)
                
                # Scrivi nel file (solo primi 5)
                if i < 5:
                    line_data += f" {m:10.6f} {q0:10.6f}"
            
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
print("GENERAZIONE PLOT...")
print("=" * 80)

# ==================== PLOT 1: SPETTRO AUTOVALORI ====================
fig, axes = plt.subplots(len(lambda_values), len(alpha_values), 
                        figsize=(6*len(alpha_values), 5*len(lambda_values)))
if len(lambda_values) == 1 and len(alpha_values) == 1:
    axes = np.array([[axes]])
elif len(lambda_values) == 1:
    axes = axes.reshape(1, -1)
elif len(alpha_values) == 1:
    axes = axes.reshape(-1, 1)

for i, lam in enumerate(lambda_values):
    for j, alpha in enumerate(alpha_values):
        ax = axes[i, j]
        key = (lam, alpha)
        results = all_results[key]
        
        # Plot spettro
        x = np.arange(len(results['eigenvalues']))
        ax.plot(x, results['eigenvalues'], 'o-', linewidth=2, markersize=6, color='steelblue')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Evidenzia i top 5
        ax.scatter(x[:5], results['eigenvalues'][:5], s=100, c='red', 
                  zorder=5, edgecolors='black', linewidth=1.5, label='Top 5')
        
        ax.set_xlabel('Eigenvalue Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Eigenvalue', fontsize=12, fontweight='bold')
        ax.set_title(f'λ={lam}, α={alpha}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

plt.suptitle('Eigenvalue Spectrum of J', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'eigenspectrum_{timestamp}.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Salvato: eigenspectrum_{}.png".format(timestamp))

# ==================== PLOT 2: MAGNETIZZAZIONE vs T (top autovettori) ====================
for lam in lambda_values:
    fig, axes = plt.subplots(1, len(alpha_values), figsize=(7*len(alpha_values), 5))
    if len(alpha_values) == 1:
        axes = [axes]
    
    for ax_idx, alpha in enumerate(alpha_values):
        ax = axes[ax_idx]
        key = (lam, alpha)
        results = all_results[key]
        
        # Plot magnetizzazione per top 5 autovettori
        n_to_plot = min(5, n_eigenvectors)
        colors = plt.cm.viridis(np.linspace(0, 1, n_to_plot))
        
        for i in range(n_to_plot):
            label = f'Eigvec {i} (λ={results["eigenvalues"][i]:.3f})'
            ax.plot(results['T'], results['magnetizations'][i],
                   'o-', linewidth=2.5, markersize=5, label=label,
                   color=colors[i], alpha=0.8)
        
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='m=0.1')
        ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
        ax.set_ylabel('Magnetization', fontsize=12, fontweight='bold')
        ax.set_title(f'α = {alpha}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)
    
    plt.suptitle(f'Eigenvector Magnetization vs Temperature | λ = {lam}',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'magnetization_lambda{lam:.2f}_{timestamp}.png',
               dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Salvato: magnetization_lambda{lam:.2f}_{timestamp}.png")

# ==================== PLOT 3: ATTIVITÀ q_0 vs T ====================
for lam in lambda_values:
    fig, axes = plt.subplots(1, len(alpha_values), figsize=(7*len(alpha_values), 5))
    if len(alpha_values) == 1:
        axes = [axes]
    
    for ax_idx, alpha in enumerate(alpha_values):
        ax = axes[ax_idx]
        key = (lam, alpha)
        results = all_results[key]
        
        # Plot attività per top 5 autovettori
        n_to_plot = min(5, n_eigenvectors)
        colors = plt.cm.plasma(np.linspace(0, 1, n_to_plot))
        
        for i in range(n_to_plot):
            label = f'Eigvec {i}'
            ax.plot(results['T'], results['q_0'][i],
                   'o-', linewidth=2, markersize=4, label=label,
                   color=colors[i], alpha=0.8)
        
        ax.axhline(y=f, color='red', linestyle='--', alpha=0.5, 
                  linewidth=2, label=f'f={f}')
        
        ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'$q_0$ (Activity)', fontsize=12, fontweight='bold')
        ax.set_title(f'α = {alpha}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.25)
    
    plt.suptitle(f'Network Activity vs Temperature | λ = {lam}',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'activity_lambda{lam:.2f}_{timestamp}.png',
               dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Salvato: activity_lambda{lam:.2f}_{timestamp}.png")

# ==================== PLOT 4: MAGNETIZZAZIONE E ATTIVITÀ INSIEME ====================
for lam in lambda_values:
    for alpha in alpha_values:
        key = (lam, alpha)
        results = all_results[key]
        
        # Plot per top 3 autovettori
        n_to_plot = min(3, n_eigenvectors)
        fig, axes = plt.subplots(2, n_to_plot, figsize=(7*n_to_plot, 8))
        
        if n_to_plot == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_to_plot):
            # Subplot magnetizzazione
            ax1 = axes[0, i]
            ax1.plot(results['T'], results['magnetizations'][i],
                    'o-', linewidth=3, markersize=6, color='steelblue', alpha=0.8)
            ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax1.set_ylabel('Magnetization', fontsize=12, fontweight='bold')
            ax1.set_title(f'Eigenvec {i} (λ={results["eigenvalues"][i]:.3f})',
                         fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.1, 1.05)
            ax1.set_xticklabels([])
            
            # Subplot attività
            ax2 = axes[1, i]
            ax2.plot(results['T'], results['q_0'][i],
                    'o-', linewidth=3, markersize=6, color='orange', alpha=0.8)
            ax2.axhline(y=f, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax2.set_xlabel('Temperature', fontsize=12, fontweight='bold')
            ax2.set_ylabel(r'$q_0$ (Activity)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 0.25)
        
        plt.suptitle(f'Magnetization & Activity | λ={lam}, α={alpha}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'combined_lambda{lam:.2f}_alpha{alpha:.2f}_{timestamp}.png',
                   dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ Salvato: combined_lambda{lam:.2f}_alpha{alpha:.2f}_{timestamp}.png")

print("\n" + "=" * 80)
print("ANALISI COMPLETATA!")
print("=" * 80)
print(f"✓ File dati: {output_filename}")
print(f"✓ Plot generati per {len(lambda_values)} λ × {len(alpha_values)} α")
print("\nINTERPRETAZIONE:")
print("- m alto a T bassa → autovettore = attrattore stabile")
print("- m scende con T → transizione di fase")
print("- q_0 ≈ f → rete mantiene attività corretta")
print("- Autovalori grandi → attrattori più forti")
print("=" * 80)
