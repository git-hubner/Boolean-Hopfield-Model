# -*- coding: utf-8 -*-
"""Simulated Annealing per trovare minimi locali e globali"""

import numpy as np
from numpy.random import rand, randn, binomial
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# ===============================
# SETUP SISTEMA
# ===============================
N = 10000
P = 500
f = 0.1
s_e = 0.7
s_t = np.sqrt(1 - s_e**2)
a = N/4
T_threshold = norm.ppf(1 - f)
lam = 1 - a/N
alpha = P / N
theta_field = 0.7 - f

# Genera pattern
I_ep = np.random.normal(0, s_e, size=(P, N))
I_t = np.zeros((P, N))
I_t[0] = s_t * np.random.normal(size=N)
eta = np.zeros((P, N))

for i in range(P):
    if i > 0:
        z_t = np.random.normal(size=N)
        I_t[i] = lam * I_t[i-1] + s_t * np.sqrt(1 - lam**2) * z_t
    for j in range(N):
        if I_t[i][j] + I_ep[i][j] - T_threshold >= 0:
            eta[i][j] = 1
        else:
            eta[i][j] = 0

# Costruisci matrice J
J = np.zeros((N, N))
eta_cent = eta - f
J = np.dot(eta_cent.T, eta_cent) / (f * (1 - f) * N)
np.fill_diagonal(J, 0)


def prob(h, b):
    """Probabilità di attivazione"""
    return 1 / (1 + np.exp(-np.clip((h - theta_field) * b, -500, 500)))


def energy(W, J):
    """
    Energia dall'Hamiltoniana: H = -1/2 Σᵢⱼ Jᵢⱼ Vᵢ Vⱼ + θ Σᵢ Vᵢ
    """
    h = np.dot(J, W)
    E_interaction = -0.5 * np.dot(W, h)
    E_field = theta_field * np.sum(W)
    return E_interaction + E_field


def compute_overlap(W, eta_target):
    """Calcola overlap con pattern target"""
    return np.dot(eta_target - f, W) / ((1 - f) * np.sum(eta_target))


def hamming_distance(W1, W2):
    """Distanza di Hamming tra due configurazioni"""
    return np.sum(W1 != W2)


# ===============================
# SIMULATED ANNEALING
# ===============================
class SimulatedAnnealing:
    """
    Simulated Annealing per trovare minimi dell'energia
    """
    def __init__(self, J, eta, T_init=2.0, T_final=0.01, cooling_rate=0.95):
        """
        Parameters:
        -----------
        J : array (N,N)
            Matrice sinaptica
        eta : array (P,N)
            Pattern
        T_init : float
            Temperatura iniziale (alta)
        T_final : float
            Temperatura finale (bassa)
        cooling_rate : float
            Fattore di raffreddamento (0 < α < 1)
        """
        self.J = J
        self.N = J.shape[0]
        self.eta = eta
        self.T_init = T_init
        self.T_final = T_final
        self.cooling_rate = cooling_rate
        
        # Storia
        self.history = {
            'temperatures': [],
            'energies': [],
            'overlaps': [],
            'activities': [],
            'configurations': []
        }
    
    def run(self, W_init, target_pattern_idx, n_steps_per_T=100, verbose=True):
        """
        Esegue simulated annealing partendo da W_init
        
        Parameters:
        -----------
        W_init : array (N,)
            Configurazione iniziale
        target_pattern_idx : int
            Indice pattern di riferimento per overlap
        n_steps_per_T : int
            Numero di step MC per ogni temperatura
        
        Returns:
        --------
        W_final : array (N,)
            Configurazione finale (minimo trovato)
        E_final : float
            Energia finale
        history : dict
            Storia dell'annealing
        """
        W = W_init.copy()
        T = self.T_init
        eta_target = self.eta[target_pattern_idx]
        
        if verbose:
            print("="*70)
            print("SIMULATED ANNEALING")
            print("="*70)
            print(f"T_init: {self.T_init:.4f}, T_final: {self.T_final:.4f}")
            print(f"Cooling rate: {self.cooling_rate}")
            print(f"Steps per T: {n_steps_per_T}")
            print()
        
        step = 0
        
        while T > self.T_final:
            beta = 1.0 / T
            
            # Monte Carlo steps a questa temperatura
            for _ in range(n_steps_per_T):
                # Proponi flip di un neurone casuale
                i = np.random.randint(self.N)
                W_old = W[i]
                
                # Calcola energia prima
                E_old = energy(W, self.J)
                
                # Flip
                W[i] = 1 - W[i]
                
                # Calcola energia dopo
                E_new = energy(W, self.J)
                
                # Accettazione Metropolis
                delta_E = E_new - E_old
                
                if delta_E <= 0 or rand() < np.exp(-beta * delta_E):
                    # Accetta
                    pass
                else:
                    # Rifiuta: ripristina
                    W[i] = W_old
                
                step += 1
            
            # Salva stato corrente
            E_current = energy(W, self.J)
            m_current = compute_overlap(W, eta_target)
            activity = np.mean(W)
            
            self.history['temperatures'].append(T)
            self.history['energies'].append(E_current)
            self.history['overlaps'].append(m_current)
            self.history['activities'].append(activity)
            self.history['configurations'].append(W.copy())
            
            if verbose and len(self.history['temperatures']) % 10 == 0:
                print(f"T={T:.4f}, E={E_current:.2f}, m={m_current:.4f}, "
                      f"activity={activity:.4f}")
            
            # Raffredda
            T *= self.cooling_rate
        
        E_final = energy(W, self.J)
        
        if verbose:
            print(f"\nAnnealing completato!")
            print(f"Energia finale: {E_final:.2f}")
            print(f"Overlap finale: {compute_overlap(W, eta_target):.4f}")
        
        return W, E_final, self.history


# ===============================
# MULTIPLE RUNS PER TROVARE DIVERSI MINIMI
# ===============================
def find_multiple_minima(J, eta, target_idx, n_runs=20, T_init=2.0, T_final=0.01):
    """
    Esegue multiple run di simulated annealing da configurazioni iniziali diverse
    per trovare diversi minimi (locali e globali)
    
    Parameters:
    -----------
    J : array (N,N)
        Matrice sinaptica
    eta : array (P,N)
        Pattern
    target_idx : int
        Indice pattern target
    n_runs : int
        Numero di run indipendenti
    
    Returns:
    --------
    minima : list of dict
        Lista di minimi trovati con {W, E, m, activity}
    """
    print("="*70)
    print(f"RICERCA MINIMI: {n_runs} run indipendenti")
    print("="*70)
    
    minima = []
    
    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        
        # Configurazione iniziale casuale (diversa per ogni run)
        if run < n_runs // 2:
            # Prima metà: partenza da configurazioni sparse (m alto)
            W_init = eta[target_idx].copy()
            flip_frac = 0.1 + 0.3 * (run / (n_runs // 2))
            flip_idx = np.random.choice(N, size=int(N * flip_frac), replace=False)
            W_init[flip_idx] = 1 - W_init[flip_idx]
        else:
            # Seconda metà: partenza casuale (esplora tutto lo spazio)
            W_init = binomial(1, 0.05 + 0.3 * rand(), size=N)
        
        # Simulated annealing
        sa = SimulatedAnnealing(J, eta, T_init=T_init, T_final=T_final, 
                               cooling_rate=0.95)
        W_final, E_final, history = sa.run(W_init, target_idx, 
                                           n_steps_per_T=50, verbose=False)
        
        # Salva minimo
        m_final = compute_overlap(W_final, eta[target_idx])
        activity_final = np.mean(W_final)
        
        minima.append({
            'W': W_final.copy(),
            'E': E_final,
            'm': m_final,
            'activity': activity_final,
            'run': run,
            'history': history
        })
        
        print(f"  → E={E_final:.2f}, m={m_final:.4f}, activity={activity_final:.4f}")
    
    # Ordina per energia
    minima.sort(key=lambda x: x['E'])
    
    return minima


# ===============================
# CLUSTERING DEI MINIMI
# ===============================
def cluster_minima(minima, distance_threshold=0.1):
    """
    Raggruppa minimi simili usando clustering gerarchico
    
    Parameters:
    -----------
    minima : list of dict
        Lista di minimi trovati
    distance_threshold : float
        Soglia per considerare due minimi nello stesso cluster
    
    Returns:
    --------
    clusters : dict
        Dizionario con cluster_id → lista di indici
    representatives : list
        Rappresentanti di ogni cluster
    """
    n_minima = len(minima)
    
    # Calcola matrice di distanze (Hamming)
    distances = np.zeros((n_minima, n_minima))
    for i in range(n_minima):
        for j in range(i+1, n_minima):
            d = hamming_distance(minima[i]['W'], minima[j]['W']) / N
            distances[i, j] = d
            distances[j, i] = d
    
    # Clustering gerarchico
    condensed_dist = squareform(distances)
    linkage_matrix = linkage(condensed_dist, method='average')
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
    # Organizza in cluster
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    
    # Trova rappresentante di ogni cluster (quello con energia più bassa)
    representatives = []
    for label, indices in clusters.items():
        energies = [minima[i]['E'] for i in indices]
        best_idx = indices[np.argmin(energies)]
        representatives.append(minima[best_idx])
    
    # Ordina per energia
    representatives.sort(key=lambda x: x['E'])
    
    return clusters, representatives


# ===============================
# ANALISI E VISUALIZZAZIONE
# ===============================
def analyze_minima(minima, clusters, representatives):
    """Analizza e visualizza i minimi trovati"""
    
    print("\n" + "="*70)
    print("ANALISI DEI MINIMI")
    print("="*70)
    
    # Identifica minimo globale
    global_min = representatives[0]
    
    print(f"\nNumero totale di run: {len(minima)}")
    print(f"Numero di cluster (bacini): {len(clusters)}")
    print(f"Numero di rappresentanti unici: {len(representatives)}")
    
    print(f"\n{'Cluster':<10} {'N. minimi':<12} {'E medio':<12} {'E min':<12} {'m medio':<12}")
    print("-"*58)
    
    for cluster_id, indices in clusters.items():
        E_vals = [minima[i]['E'] for i in indices]
        m_vals = [minima[i]['m'] for i in indices]
        print(f"{cluster_id:<10} {len(indices):<12} {np.mean(E_vals):<12.2f} "
              f"{np.min(E_vals):<12.2f} {np.mean(m_vals):<12.4f}")
    
    print(f"\n{'='*70}")
    print("MINIMI RAPPRESENTATIVI (uno per cluster):")
    print("="*70)
    print(f"{'Tipo':<15} {'Energia':<12} {'Overlap m':<12} {'Attività':<12} {'ΔE dal globale':<15}")
    print("-"*66)
    
    for i, rep in enumerate(representatives):
        tipo = "GLOBALE" if i == 0 else f"LOCALE {i}"
        delta_E = rep['E'] - global_min['E']
        print(f"{tipo:<15} {rep['E']:<12.2f} {rep['m']:<12.4f} "
              f"{rep['activity']:<12.4f} {delta_E:<15.2f}")
    
    # ===============================
    # PLOT
    # ===============================
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Energia di tutti i minimi
    ax1 = plt.subplot(3, 3, 1)
    energies = [m['E'] for m in minima]
    ax1.hist(energies, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(global_min['E'], color='red', linestyle='--', linewidth=2, 
               label='Minimo globale')
    ax1.set_xlabel('Energia', fontsize=11)
    ax1.set_ylabel('Frequenza', fontsize=11)
    ax1.set_title('Distribuzione Energie')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Overlap vs Energia
    ax2 = plt.subplot(3, 3, 2)
    overlaps = [m['m'] for m in minima]
    ax2.scatter(energies, overlaps, alpha=0.6, s=50)
    
    # Evidenzia rappresentanti
    rep_E = [r['E'] for r in representatives]
    rep_m = [r['m'] for r in representatives]
    ax2.scatter(rep_E, rep_m, c='red', s=200, marker='*', 
               edgecolors='black', linewidth=2, label='Rappresentanti', zorder=10)
    
    ax2.set_xlabel('Energia', fontsize=11)
    ax2.set_ylabel('Overlap m', fontsize=11)
    ax2.set_title('Overlap vs Energia')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Attività vs Energia
    ax3 = plt.subplot(3, 3, 3)
    activities = [m['activity'] for m in minima]
    ax3.scatter(energies, activities, alpha=0.6, s=50)
    
    rep_act = [r['activity'] for r in representatives]
    ax3.scatter(rep_E, rep_act, c='red', s=200, marker='*', 
               edgecolors='black', linewidth=2, label='Rappresentanti', zorder=10)
    ax3.axhline(f, color='gray', linestyle='--', alpha=0.5, label=f'f={f}')
    
    ax3.set_xlabel('Energia', fontsize=11)
    ax3.set_ylabel('Attività', fontsize=11)
    ax3.set_title('Attività vs Energia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Barriere energetiche (storia annealing per alcuni run)
    ax4 = plt.subplot(3, 3, 4)
    
    for i in [0, len(minima)//4, len(minima)//2, -1]:
        history = minima[i]['history']
        ax4.plot(history['temperatures'], history['energies'], 
                alpha=0.7, linewidth=2, label=f"Run {minima[i]['run']}")
    
    ax4.set_xlabel('Temperatura', fontsize=11)
    ax4.set_ylabel('Energia', fontsize=11)
    ax4.set_title('Traiettorie Annealing')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Dimensione dei cluster
    ax5 = plt.subplot(3, 3, 5)
    cluster_sizes = [len(indices) for indices in clusters.values()]
    cluster_ids = list(clusters.keys())
    
    ax5.bar(range(len(cluster_sizes)), cluster_sizes, alpha=0.7, color='green')
    ax5.set_xlabel('Cluster ID', fontsize=11)
    ax5.set_ylabel('Numero di minimi', fontsize=11)
    ax5.set_title('Dimensione Bacini di Attrazione')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Matrice di distanze tra rappresentanti
    ax6 = plt.subplot(3, 3, 6)
    
    n_rep = len(representatives)
    dist_matrix = np.zeros((n_rep, n_rep))
    
    for i in range(n_rep):
        for j in range(n_rep):
            dist_matrix[i, j] = hamming_distance(representatives[i]['W'], 
                                                 representatives[j]['W']) / N
    
    im = ax6.imshow(dist_matrix, cmap='viridis', aspect='auto')
    ax6.set_xlabel('Rappresentante', fontsize=11)
    ax6.set_ylabel('Rappresentante', fontsize=11)
    ax6.set_title('Distanze (Hamming) tra Minimi')
    plt.colorbar(im, ax=ax6, label='Distanza normalizzata')
    
    # 7. Profilo energetico lungo dimensione principale
    ax7 = plt.subplot(3, 3, 7)
    
    # Ordina per overlap
    sorted_indices = np.argsort(overlaps)
    sorted_E = np.array(energies)[sorted_indices]
    sorted_m = np.array(overlaps)[sorted_indices]
    
    ax7.plot(sorted_m, sorted_E, 'o-', alpha=0.5, markersize=3)
    ax7.scatter(rep_m, rep_E, c='red', s=200, marker='*', 
               edgecolors='black', linewidth=2, zorder=10)
    ax7.set_xlabel('Overlap m', fontsize=11)
    ax7.set_ylabel('Energia', fontsize=11)
    ax7.set_title('Profilo Energetico vs m')
    ax7.grid(True, alpha=0.3)
    
    # 8. Convergenza dell'energia (storia per minimo globale)
    ax8 = plt.subplot(3, 3, 8)
    
    # Trova quale run ha portato al minimo globale
    global_run_idx = global_min['run']
    for m in minima:
        if m['run'] == global_run_idx:
            history_global = m['history']
            break
    
    steps = range(len(history_global['energies']))
    ax8.plot(steps, history_global['energies'], 'b-', linewidth=2)
    ax8.axhline(global_min['E'], color='red', linestyle='--', 
               linewidth=2, label='E finale')
    ax8.set_xlabel('Step', fontsize=11)
    ax8.set_ylabel('Energia', fontsize=11)
    ax8.set_title(f'Convergenza al Minimo Globale (Run {global_run_idx})')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Overlap nel tempo per minimo globale
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(steps, history_global['overlaps'], 'g-', linewidth=2)
    ax9.set_xlabel('Step', fontsize=11)
    ax9.set_ylabel('Overlap m', fontsize=11)
    ax9.set_title('Evoluzione Overlap (Minimo Globale)')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'simulated_annealing_minima_alpha={alpha:.2f}.png', dpi=150)
    plt.show()


# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    target_idx = 100
    
    # Trova multipli minimi
    minima = find_multiple_minima(J, eta, target_idx, n_runs=30, 
                                  T_init=2.0, T_final=0.001)
    
    # Clustering
    clusters, representatives = cluster_minima(minima, distance_threshold=0.15)
    
    # Analisi
    analyze_minima(minima, clusters, representatives)
    
    # Salva configurazioni rappresentative
    print("\n" + "="*70)
    print("SALVATAGGIO CONFIGURAZIONI RAPPRESENTATIVE")
    print("="*70)
    
    np.savez(f'minima_representatives_alpha={alpha:.2f}.npz',
             representatives=[r['W'] for r in representatives],
             energies=[r['E'] for r in representatives],
             overlaps=[r['m'] for r in representatives],
             activities=[r['activity'] for r in representatives])
    
    print(f"Salvate {len(representatives)} configurazioni rappresentative")
    print("File: minima_representatives_alpha={alpha:.2f}.npz")
