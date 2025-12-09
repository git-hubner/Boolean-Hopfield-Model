# -*- coding: utf-8 -*-
"""Parallel Tempering alla SECONDA transizione di fase (SG → Para)"""

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
theta_field = 0.7 - f  # Campo esterno dalla prob()

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


def compute_overlap(W, eta_target):
    """Calcola overlap con pattern target"""
    return np.dot(eta_target - f, W) / ((1 - f) * np.sum(eta_target))


# ===============================
# FASE 1: TROVA ENTRAMBE LE TRANSIZIONI
# ===============================
def find_both_transitions(J, eta, target_idx, T_range=(0.01, 0.8), n_points=40):
    """
    Trova ENTRAMBE le temperature di transizione:
    1. T_c1: Retrieval → Spin Glass (m: 1 → 0.3)
    2. T_c2: Spin Glass → Paramagnetica (m: 0.3 → 0)
    
    Returns:
    --------
    T_c1, T_c2 : float
        Temperature critiche
    """
    print("="*70)
    print("FASE 1: Ricerca delle DUE temperature critiche")
    print("="*70)
    
    T_values = np.linspace(T_range[0], T_range[1], n_points)
    m_values = []
    
    W = eta[target_idx].copy()
    flip_idx = np.random.choice(N, size=N // 20, replace=False)
    W[flip_idx] = 1 - W[flip_idx]
    
    for T in T_values:
        beta = 1.0 / T
        
        # Termalizza più a lungo per esplorare bene
        for _ in range(300):
            for i in np.random.permutation(N):
                h = np.dot(J[i], W)
                phi = prob(h, beta)
                W[i] = 1 if rand() < phi else 0
        
        # Misura overlap (media su più campioni)
        m_samples = []
        for _ in range(10):
            for i in np.random.permutation(N):
                h = np.dot(J[i], W)
                phi = prob(h, beta)
                W[i] = 1 if rand() < phi else 0
            m_samples.append(compute_overlap(W, eta[target_idx]))
        
        m = np.mean(m_samples)
        m_values.append(m)
        
        if len(m_values) > 1:
            print(f"T = {T:.4f}: m = {m:.4f}, Δm = {m - m_values[-2]:.4f}")
        else:
            print(f"T = {T:.4f}: m = {m:.4f}")
    
    m_array = np.array(m_values)
    
    # Trova T_c1: prima transizione (m scende sotto 0.7)
    transition1_indices = np.where(m_array < 0.7)[0]
    if len(transition1_indices) > 0:
        T_c1_idx = transition1_indices[0]
        T_c1 = T_values[T_c1_idx]
    else:
        T_c1 = T_range[0]
    
    # Trova T_c2: seconda transizione (m scende sotto 0.15)
    transition2_indices = np.where(m_array < 0.15)[0]
    if len(transition2_indices) > 0:
        T_c2_idx = transition2_indices[0]
        T_c2 = T_values[T_c2_idx]
    else:
        T_c2 = T_range[1]
    
    print(f"\n>>> PRIMA transizione (Retrieval → SG): T_c1 ≈ {T_c1:.4f}")
    print(f">>> SECONDA transizione (SG → Para):    T_c2 ≈ {T_c2:.4f}")
    print(f">>> Useremo T_c2 per il Parallel Tempering")
    
    return T_c1, T_c2, T_values, m_values


# ===============================
# FASE 2: PARALLEL TEMPERING
# ===============================
class ParallelTempering:
    """
    Implementazione di Parallel Tempering (Replica Exchange Monte Carlo)
    con energia corretta dall'Hamiltoniana
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
        
        # Inizializza repliche nella fase spin glass (m piccolo)
        self.replicas = []
        for _ in range(n_replicas):
            # Partenza casuale per fase SG
            W_init = binomial(1, 0.15, size=N)  # Attività bassa
            self.replicas.append(W_init)
        
        # Statistiche
        self.swap_attempts = 0
        self.swap_accepts = 0
        
        print(f"\nParallel Tempering inizializzato:")
        print(f"  Repliche: {n_replicas}")
        print(f"  Temperature: {self.temperatures}")
    
    def energy(self, W):
        """
        Calcola energia corretta dall'Hamiltoniana:
        H = -1/2 Σᵢⱼ Jᵢⱼ Vᵢ Vⱼ + θ Σᵢ Vᵢ
        
        dove θ = 0.7 - f è il campo esterno
        """
        # Interazione sinaptica: -1/2 Σᵢⱼ Jᵢⱼ Vᵢ Vⱼ
        # Possiamo scriverla come: -1/2 Σᵢ Vᵢ hᵢ dove hᵢ = Σⱼ Jᵢⱼ Vⱼ
        h = np.dot(self.J, W)
        E_interaction = -0.5 * np.dot(W, h)
        
        # Campo esterno: θ Σᵢ Vᵢ
        E_field = theta_field * np.sum(W)
        
        return E_interaction + E_field
    
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
        """Tenta scambio tra replica i e j usando energia corretta"""
        beta_i = self.betas[i]
        beta_j = self.betas[j]
        
        E_i = self.energy(self.replicas[i])
        E_j = self.energy(self.replicas[j])
        
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
        print("FASE 2: Parallel Tempering alla SECONDA transizione (SG → Para)")
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
                # Schema alternato: prima pari-dispari, poi dispari-pari
                if (step // swap_interval) % 2 == 0:
                    pairs = range(0, self.n_replicas - 1, 2)
                else:
                    pairs = range(1, self.n_replicas - 1, 2)
                
                for i in pairs:
                    self.attempt_swap(i, i + 1)
            
            # Misura osservabili
            if step % measure_interval == 0:
                measurements['step'].append(step)
                
                for replica_idx in range(self.n_replicas):
                    W = self.replicas[replica_idx]
                    
                    # Overlap
                    m = compute_overlap(W, self.eta_target)
                    measurements['overlaps'][replica_idx].append(m)
                    
                    # Energia
                    E = self.energy(W)
                    measurements['energies'][replica_idx].append(E)
                    
                    # Attività
                    activity = np.mean(W)
                    measurements['activities'][replica_idx].append(activity)
                
                if step % (10 * measure_interval) == 0:
                    swap_rate = self.swap_accepts/(self.swap_attempts+1e-10)
                    m_low_T = measurements['overlaps'][0][-1]
                    m_high_T = measurements['overlaps'][-1][-1]
                    print(f"Step {step}/{n_steps} - Swap: {swap_rate:.3f}, "
                          f"m(T_min)={m_low_T:.3f}, m(T_max)={m_high_T:.3f}")
        
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
    ax1.set_title('Overlap vs Step (SG → Para)')
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
    ax3.axhline(f, color='gray', linestyle='--', alpha=0.5, label=f'f={f}')
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
    ax5.set_title('Overlap medio vs T (transizione SG→Para)')
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
    ax7.set_title('Calore Specifico (picco = transizione)')
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
    ax8.set_title('Suscettività (picco = transizione)')
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
    plt.savefig(f'parallel_tempering_SG_Para_alpha={alpha:.2f}.png', dpi=150)
    plt.show()
    
    # Stampa statistiche
    print("\n" + "="*70)
    print("STATISTICHE ALL'EQUILIBRIO (Transizione SG → Para):")
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
    
    # Fase 1: Trova ENTRAMBE le transizioni
    T_c1, T_c2, T_scan, m_scan = find_both_transitions(J, eta, target_idx)
    
    # Plot scan iniziale con entrambe le transizioni
    plt.figure(figsize=(10, 6))
    plt.plot(T_scan, m_scan, 'bo-', linewidth=2, markersize=6)
    plt.axvline(T_c1, color='orange', linestyle='--', linewidth=2, label=f'T_c1 ≈ {T_c1:.4f} (Ret→SG)')
    plt.axvline(T_c2, color='red', linestyle='--', linewidth=2, label=f'T_c2 ≈ {T_c2:.4f} (SG→Para)')
    plt.axhspan(0.6, 1.0, alpha=0.1, color='blue', label='Retrieval')
    plt.axhspan(0.1, 0.4, alpha=0.1, color='green', label='Spin Glass')
    plt.axhspan(0.0, 0.1, alpha=0.1, color='red', label='Paramagnetica')
    plt.xlabel('Temperatura', fontsize=12)
    plt.ylabel('Overlap m', fontsize=12)
    plt.title('Scan completo: due transizioni di fase')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('temperature_scan_two_transitions.png', dpi=150)
    plt.show()
    
    # Fase 2: Parallel Tempering intorno a T_c2 (SECONDA TRANSIZIONE)
    T_min = max(0.01, T_c2 - 0.08)
    T_max = T_c2 + 0.08
    
    print(f"\nFocus su SECONDA transizione:")
    print(f"Range PT: [{T_min:.4f}, {T_max:.4f}]")
    
    pt = ParallelTempering(J, eta, target_idx, T_min, T_max, n_replicas=10)
    measurements = pt.run(n_steps=3000, swap_interval=10, measure_interval=5)
    
    # Analisi risultati
    analyze_measurements(measurements)
