# -*- coding: utf-8 -*-
"""Parallel Tempering alla transizione di fase"""

import numpy as np
from numpy.random import rand, randn, randint, normal, binomial
import matplotlib.pyplot as plt
from scipy.stats import norm

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
    return 1 / (1 + np.exp(-np.clip((h - (0.7 - f)) * b, -500, 500)))


def compute_overlap(W, eta_target):
    """Calcola overlap con pattern target"""
    return np.dot(eta_target - f, W) / ((1 - f) * np.sum(eta_target))


# ===============================
# FASE 1: TROVA TEMPERATURA CRITICA
# ===============================
def find_transition_temperature(J, eta, target_idx, T_range=(0.05, 0.3), n_points=20):
    """
    Trova la temperatura di transizione (dove m crolla)
    
    Returns:
    --------
    T_c : float
        Temperatura critica stimata
    """
    print("="*70)
    print("FASE 1: Ricerca temperatura critica")
    print("="*70)
    
    T_values = np.linspace(T_range[0], T_range[1], n_points)
    m_values = []
    
    W = eta[target_idx].copy()
    flip_idx = np.random.choice(N, size=N // 20, replace=False)
    W[flip_idx] = 1 - W[flip_idx]
    
    for T in T_values:
        beta = 1.0 / T
        
        # Termalizza
        for _ in range(200):
            for i in np.random.permutation(N):
                h = np.dot(J[i], W)
                phi = prob(h, beta)
                W[i] = 1 if rand() < phi else 0
        
        # Misura overlap
        m = compute_overlap(W, eta[target_idx])
        m_values.append(m)
        
        print(f"T = {T:.4f}: m = {m:.4f}")
    
    # Trova T_c come punto dove m scende sotto 0.7
    m_array = np.array(m_values)
    transition_indices = np.where(m_array < 0.7)[0]
    
    if len(transition_indices) > 0:
        T_c_idx = transition_indices[0]
        T_c = T_values[T_c_idx]
        print(f"\n>>> Temperatura critica stimata: T_c ≈ {T_c:.4f}")
    else:
        T_c = T_range[1]
        print(f"\n>>> Nessuna transizione trovata, uso T = {T_c:.4f}")
    
    return T_c, T_values, m_values


# ===============================
# FASE 2: PARALLEL TEMPERING
# ===============================
class ParallelTempering:
    """
    Implementazione di Parallel Tempering (Replica Exchange Monte Carlo)
    """
    def __init__(self, J, eta, target_idx, T_min, T_max, n_replicas=8):
        """
        Parameters:
        -----------
        J : array (N,N)
            Matrice sinaptica
        eta : array (P,N)
            Pattern
        target_idx : int
            Indice pattern target
        T_min, T_max : float
            Range di temperature
        n_replicas : int
            Numero di repliche
        """
        self.J = J
        self.N = J.shape[0]
        self.eta = eta
        self.target_idx = target_idx
        self.eta_target = eta[target_idx]
        
        # Temperature geometricamente spaziate
        self.temperatures = np.geomspace(T_min, T_max, n_replicas)
        self.betas = 1.0 / self.temperatures
        self.n_replicas = n_replicas
        
        # Inizializza repliche
        self.replicas = []
        W_init = eta[target_idx].copy()
        flip_idx = np.random.choice(self.N, size=self.N // 20, replace=False)
        W_init[flip_idx] = 1 - W_init[flip_idx]
        
        for _ in range(n_replicas):
            self.replicas.append(W_init.copy())
        
        # Statistiche
        self.swap_attempts = 0
        self.swap_accepts = 0
        
        print(f"\nParallel Tempering inizializzato:")
        print(f"  Repliche: {n_replicas}")
        print(f"  Temperature: {self.temperatures}")
    
    def energy(self, W, beta):
        """Calcola energia di una configurazione"""
        h = np.dot(self.J, W)
        E = -np.sum(W * (h- (0.7-f))) / 2
        return E
    
    def mc_sweep(self, replica_idx, n_sweeps=1):
        """Esegue sweep Monte Carlo su una replica"""
        W = self.replicas[replica_idx]
        beta = self.betas[replica_idx]
        
        for _ in range(n_sweeps):
            for i in np.random.permutation(self.N):
                h = np.dot(self.J[i], W)
                phi = prob(h, beta)
                W[i] = 1 if rand() < phi else 0
    
    def attempt_swap(self, i, j):
        """Tenta scambio tra replica i e j"""
        beta_i = self.betas[i]
        beta_j = self.betas[j]
        
        E_i = self.energy(self.replicas[i], beta_i)
        E_j = self.energy(self.replicas[j], beta_j)
        
        # Criterio Metropolis per lo swap
        delta = (beta_j - beta_i) * (E_i - E_j)
        
        self.swap_attempts += 1
        
        if delta <= 0 or rand() < np.exp(-delta):
            # Accetta swap
            self.replicas[i], self.replicas[j] = self.replicas[j], self.replicas[i]
            self.swap_accepts += 1
            return True
        
        return False
    
    def run(self, n_steps=1000, swap_interval=10, measure_interval=10):
        """
        Esegue parallel tempering
        
        Parameters:
        -----------
        n_steps : int
            Numero di step totali
        swap_interval : int
            Ogni quanti step tentare swap
        measure_interval : int
            Ogni quanti step misurare osservabili
        
        Returns:
        --------
        measurements : dict
            Dizionario con misure per ogni replica
        """
        print(f"\n{'='*70}")
        print("FASE 2: Parallel Tempering")
        print(f"{'='*70}")
        print(f"Steps: {n_steps}, Swap ogni: {swap_interval}, Misure ogni: {measure_interval}")
        
        # Dizionari per salvare misure
        measurements = {
            'temperatures': self.temperatures,
            'overlaps': [[] for _ in range(self.n_replicas)],
            'energies': [[] for _ in range(self.n_replicas)],
            'activities': [[] for _ in range(self.n_replicas)],
            'step': []
        }
        
        for step in range(n_steps):
            # MC sweep su tutte le repliche
            for replica_idx in range(self.n_replicas):
                self.mc_sweep(replica_idx, n_sweeps=1)
            
            # Tenta swap tra repliche adiacenti
            if step % swap_interval == 0:
                for i in range(self.n_replicas - 1):
                    if rand() < 0.5:  # Swap stocastico
                        self.attempt_swap(i, i + 1)
            
            # Misura osservabili
            if step % measure_interval == 0:
                measurements['step'].append(step)
                
                for replica_idx in range(self.n_replicas):
                    W = self.replicas[replica_idx]
                    beta = self.betas[replica_idx]
                    
                    # Overlap
                    m = compute_overlap(W, self.eta_target)
                    measurements['overlaps'][replica_idx].append(m)
                    
                    # Energia
                    E = self.energy(W, beta)
                    measurements['energies'][replica_idx].append(E)
                    
                    # Attività
                    activity = np.mean(W)
                    measurements['activities'][replica_idx].append(activity)
                
                if step % (10 * measure_interval) == 0:
                    print(f"Step {step}/{n_steps} - Swap rate: {self.swap_accepts/(self.swap_attempts+1e-10):.3f}")
        
        print(f"\nParallel Tempering completato!")
        print(f"Swap acceptance rate: {self.swap_accepts/self.swap_attempts:.3f}")
        
        return measurements


def analyze_measurements(measurements):
    """Analizza e plotta risultati del parallel tempering"""
    temperatures = measurements['temperatures']
    n_replicas = len(temperatures)
    steps = measurements['step']
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Overlap vs step per ogni temperatura
    ax1 = plt.subplot(3, 3, 1)
    for i in range(n_replicas):
        ax1.plot(steps, measurements['overlaps'][i], alpha=0.7, 
                label=f'T={temperatures[i]:.3f}')
    ax1.set_xlabel('MC Step')
    ax1.set_ylabel('Overlap m')
    ax1.set_title('Overlap vs Step')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 2. Energia vs step
    ax2 = plt.subplot(3, 3, 2)
    for i in range(n_replicas):
        ax2.plot(steps, measurements['energies'][i], alpha=0.7, 
                label=f'T={temperatures[i]:.3f}')
    ax2.set_xlabel('MC Step')
    ax2.set_ylabel('Energia')
    ax2.set_title('Energia vs Step')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # 3. Attività vs step
    ax3 = plt.subplot(3, 3, 3)
    for i in range(n_replicas):
        ax3.plot(steps, measurements['activities'][i], alpha=0.7, 
                label=f'T={temperatures[i]:.3f}')
    ax3.set_xlabel('MC Step')
    ax3.set_ylabel('Attività')
    ax3.set_title('Attività vs Step')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribuzione overlap (ultima metà dei dati)
    ax4 = plt.subplot(3, 3, 4)
    halfway = len(steps) // 2
    for i in range(n_replicas):
        overlaps_eq = measurements['overlaps'][i][halfway:]
        ax4.hist(overlaps_eq, bins=20, alpha=0.5, label=f'T={temperatures[i]:.3f}')
    ax4.set_xlabel('Overlap m')
    ax4.set_ylabel('Frequenza')
    ax4.set_title('Distribuzione Overlap (equilibrio)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Media e std overlap vs T
    ax5 = plt.subplot(3, 3, 5)
    m_means = [np.mean(measurements['overlaps'][i][halfway:]) for i in range(n_replicas)]
    m_stds = [np.std(measurements['overlaps'][i][halfway:]) for i in range(n_replicas)]
    ax5.errorbar(temperatures, m_means, yerr=m_stds, marker='o', capsize=5, linewidth=2)
    ax5.set_xlabel('Temperatura')
    ax5.set_ylabel('⟨m⟩ ± σ')
    ax5.set_title('Overlap medio vs Temperatura')
    ax5.grid(True, alpha=0.3)
    
    # 6. Energia media vs T
    ax6 = plt.subplot(3, 3, 6)
    E_means = [np.mean(measurements['energies'][i][halfway:]) for i in range(n_replicas)]
    E_stds = [np.std(measurements['energies'][i][halfway:]) for i in range(n_replicas)]
    ax6.errorbar(temperatures, E_means, yerr=E_stds, marker='s', capsize=5, linewidth=2)
    ax6.set_xlabel('Temperatura')
    ax6.set_ylabel('⟨E⟩ ± σ')
    ax6.set_title('Energia media vs Temperatura')
    ax6.grid(True, alpha=0.3)
    
    # 7. Calore specifico C = β²(⟨E²⟩ - ⟨E⟩²)
    ax7 = plt.subplot(3, 3, 7)
    C_values = []
    for i in range(n_replicas):
        E_data = measurements['energies'][i][halfway:]
        E_mean = np.mean(E_data)
        E2_mean = np.mean(np.array(E_data)**2)
        beta = 1.0 / temperatures[i]
        C = beta**2 * (E2_mean - E_mean**2) / N
        C_values.append(C)
    ax7.plot(temperatures, C_values, 'ro-', linewidth=2, markersize=8)
    ax7.set_xlabel('Temperatura')
    ax7.set_ylabel('C/N')
    ax7.set_title('Calore Specifico')
    ax7.grid(True, alpha=0.3)
    
    # 8. Suscettività χ = β(⟨m²⟩ - ⟨m⟩²)
    ax8 = plt.subplot(3, 3, 8)
    chi_values = []
    for i in range(n_replicas):
        m_data = measurements['overlaps'][i][halfway:]
        m_mean = np.mean(m_data)
        m2_mean = np.mean(np.array(m_data)**2)
        beta = 1.0 / temperatures[i]
        chi = beta * (m2_mean - m_mean**2) * N
        chi_values.append(chi)
    ax8.plot(temperatures, chi_values, 'bo-', linewidth=2, markersize=8)
    ax8.set_xlabel('Temperatura')
    ax8.set_ylabel('χ')
    ax8.set_title('Suscettività')
    ax8.grid(True, alpha=0.3)
    
    # 9. Autocorrelazione overlap (per T critica)
    ax9 = plt.subplot(3, 3, 9)
    critical_replica = n_replicas // 2  # Temperatura centrale
    m_data = np.array(measurements['overlaps'][critical_replica][halfway:])
    max_lag = min(100, len(m_data) // 2)
    autocorr = [np.corrcoef(m_data[:-lag], m_data[lag:])[0, 1] if lag > 0 
                else 1.0 for lag in range(max_lag)]
    ax9.plot(range(max_lag), autocorr, 'g-', linewidth=2)
    ax9.set_xlabel('Lag')
    ax9.set_ylabel('Autocorrelazione')
    ax9.set_title(f'Autocorr. m (T={temperatures[critical_replica]:.3f})')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'parallel_tempering_analysis_alpha={alpha:.2f}.png', dpi=150)
    plt.show()
    
    # Stampa statistiche
    print("\n" + "="*70)
    print("STATISTICHE ALL'EQUILIBRIO:")
    print("="*70)
    print(f"{'Temperatura':<12} {'⟨m⟩':<10} {'σ(m)':<10} {'⟨E⟩':<12} {'C/N':<10} {'χ':<10}")
    print("-"*70)
    for i in range(n_replicas):
        print(f"{temperatures[i]:<12.4f} {m_means[i]:<10.4f} {m_stds[i]:<10.4f} "
              f"{E_means[i]:<12.2f} {C_values[i]:<10.4f} {chi_values[i]:<10.2f}")


# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    target_idx = 100
    
    # Fase 1: Trova T_c
    T_c, T_scan, m_scan = find_transition_temperature(J, eta, target_idx)
    
    # Plot scan iniziale
    plt.figure(figsize=(10, 6))
    plt.plot(T_scan, m_scan, 'bo-', linewidth=2, markersize=8)
    plt.axvline(T_c, color='r', linestyle='--', label=f'T_c ≈ {T_c:.4f}')
    plt.xlabel('Temperatura', fontsize=12)
    plt.ylabel('Overlap m', fontsize=12)
    plt.title('Scan preliminare per trovare T_c')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('temperature_scan.png', dpi=150)
    plt.show()
    
    # Fase 2: Parallel Tempering intorno a T_c
    T_min = max(0.01, T_c - 0.05)
    T_max = T_c + 0.05
    
    pt = ParallelTempering(J, eta, target_idx, T_min, T_max, n_replicas=8)
    measurements = pt.run(n_steps=2000, swap_interval=10, measure_interval=5)
    
    # Analisi risultati
    analyze_measurements(measurements)
