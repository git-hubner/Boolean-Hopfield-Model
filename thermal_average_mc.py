# -*- coding: utf-8 -*-
"""montecarlo selfcons con r, r_0 ed energia libera"""

import numpy as np
from math import exp, gamma, log, sqrt, cos, sin, pi, erfc, tanh
from numpy.random import rand, randn, randint, uniform, normal, binomial
import matplotlib.pyplot as plt
from scipy.stats import norm

N = 10000  # numero di neuroni della rete
P = 500   # numero di pattern
f = 0.1
s_e = 0.7  # deviazione standard episodi
s_t = np.sqrt(1 - s_e**2)
a = N/4
T_threshold = norm.ppf(1 - f)
lam = 1 - a/N
alpha = P / N

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

J = np.zeros((N, N))
eta_cent = eta - f
J = np.dot(eta_cent.T, eta_cent) / (f * (1 - f) * N)
np.fill_diagonal(J, 0)


def prob(h, b):
    """Probabilità di attivazione"""
    return 1 / (1 + np.exp(-np.clip((h - (0.7 - f)) * b, -500, 500)))

def monte_carlo_sampling(W_init, J, T, n_samples=1000, n_thermalization=500):
    N = len(W_init)
    beta = 1.0 / T
    W = W_init.copy()
    samples = []

    # Termalizzazione
    for _ in range(n_thermalization):
        for i in np.random.permutation(N):
            h = np.dot(J[i], W)
            phi = prob(h, beta)
            W[i] = 1 if rand() < phi else 0

    # Campionamento
    for _ in range(n_samples):
        # Sweep completo
        for i in np.random.permutation(N):
            h = np.dot(J[i], W)
            phi = prob(h, beta)
            W[i] = 1 if rand() < phi else 0

        # Salva configurazione
        samples.append(W.copy())

    return samples


def order_params_complete(W_init, J, T, eta, target_pattern_idx, n_samples=500, n_thermalization=30):
    """
    Calcola tutti i parametri d'ordine inclusi r, r_0 ed energia libera
    
    Parameters:
    -----------
    W_init : array (N,)
        Configurazione iniziale
    J : array (N,N)
        Matrice sinaptica
    T : float
        Temperatura
    eta : array (P, N)
        Tutti i pattern
    target_pattern_idx : int
        Indice del pattern target
    n_samples : int
        Numero di campioni MC
    n_thermalization : int
        Sweep di termalizzazione
    
    Returns:
    --------
    fr : float
        Energia libera
    m : float
        Overlap con pattern target
    q0 : float
        <V_i>
    q : float
        Correlazione tra campioni successivi
    r : float
        Overlap autocorrelazione
    r0 : float
        Overlap con pattern non-target
    """
    N = len(W_init)
    P_total = eta.shape[0]
    beta = 1.0 / T
    p = f
    
    eta0 = eta[target_pattern_idx]
    
    # Genera campioni termici
    samples = monte_carlo_sampling(W_init, J, T, n_samples, n_thermalization)
    
    # ==========================================
    # Calcola m e q0 (medie termiche semplici)
    # ==========================================
    m_samples = []
    q0_samples = []
    
    for W in samples:
        m_sample = np.dot(eta0 - f, W) / ((1 - f) * np.sum(eta0))
        m_samples.append(m_sample)
        
        q0_sample = np.mean(W)
        q0_samples.append(q0_sample)
    
    m = np.mean(m_samples)
    q0 = np.mean(q0_samples)
    
    # ==========================================
    # Calcola q come correlazione tra campioni successivi
    # ==========================================
    q_alt = 0
    for i in range(len(samples) - 1):
        q_alt += np.dot(samples[i], samples[i+1]) / N
    q = q_alt / (len(samples) - 1)
    
    # ==========================================
    # Calcola r come autocorrelazione tra campioni successivi
    # (stesso metodo di q, ma con overlap invece di dot product)
    # ==========================================
    r_samples = []
    for i in range(len(samples) - 1):
        # Overlap del campione i con se stesso al tempo successivo
        overlap_consecutive = np.dot(eta0 - f, samples[i]) / ((1 - f) * np.sum(eta0)) * \
                             np.dot(eta0 - f, samples[i+1]) / ((1 - f) * np.sum(eta0))
        r_samples.append(overlap_consecutive)
    
    r = np.mean(r_samples)
    
    # ==========================================
    # Calcola r0 = overlap con pattern non-target
    # r0 = 1/(P-1) * Σ_{μ≠target} m_μ²
    # ==========================================
    m_other_patterns = []
    
    for mu in range(P_total):
        if mu == target_pattern_idx:
            continue
        
        # Calcola overlap con pattern μ per tutti i campioni
        eta_mu = eta[mu]
        m_mu_samples = []
        
        for W in samples:
            m_mu = np.dot(eta_mu - f, W) / ((1 - f) * np.sum(eta_mu))
            m_mu_samples.append(m_mu)
        
        # Media termica di m_μ
        m_mu_avg = np.mean(m_mu_samples)
        m_other_patterns.append(m_mu_avg**2)
    
    r0 = np.mean(m_other_patterns)  # Media su tutti i pattern non-target
    
    # ==========================================
    # Calcola energia libera
    # ==========================================
    z = np.random.randn(10000)  # Campioni per integrazione
    
    # Campo H
    H = (np.sqrt(alpha * r / beta) * z 
         - T_threshold 
         + ((alpha * p / 2) * (r0 - r)))
         
    log_part = np.mean(-p * np.log(1 + np.exp(beta * (H+((1-p)*m)))) + (1-p) * np.log(1 + np.exp(beta * (H-(p*m)))))
    
    # Energia libera
    fr = (-1/beta * log_part
          + m**2 / 2 
          + alpha/(2*beta) * np.log(1 - p * beta * (q0 - q) + 1e-12)
          - alpha / 2 * (q * p) / (1 - p * beta * (q0 - q) + 1e-12) 
          + alpha * (p / 2) * (r0 * q0 - r * q))
          
    E = -log_part
    
    return fr, m, q0, q, r, r0, E


# ===============================
# TEST E SIMULAZIONE
# ===============================

# Configurazione iniziale
target_idx = 100
W = eta[target_idx].copy()
flip_idx = np.random.choice(N, size=N // 20, replace=False)
W[flip_idx] = 1 - W[flip_idx]

print(f"Overlap iniziale: {np.dot(eta[target_idx]-f, W) / ((1-f)*np.sum(eta[target_idx])):.4f}")
print("\nCalcolo parametri d'ordine completi con Monte Carlo...")
print("="*70)

T_values = np.linspace(0.001, 0.4, 10)
m_vals, f_vals, q_vals, q0_vals, r_vals, r0_vals, E_vals = [], [], [], [], [], [], []

for idx, t in enumerate(T_values):
    print(f"T = {t:.3f} ({idx+1}/{len(T_values)})...", end=" ")
    
    # Calcola tutti i parametri
    fr, m, q0, q, r, r0, E = order_params_complete(W, J, t, eta, target_idx,
                                                n_samples=200, n_thermalization=100)
    
    f_vals.append(fr)
    m_vals.append(m)
    q_vals.append(q)
    q0_vals.append(q0)
    r_vals.append(r)
    r0_vals.append(r0)
    E_vals.append(E)
    
    print(f"m={m:.4f}, q={q:.4f}, q0={q0:.4f}, r={r:.4f}, r0={r0:.4f}, f={fr:.4f}, E={E:.4f}")
    
    # Aggiorna W per la prossima temperatura (continuità)
    W = monte_carlo_sampling(W, J, t, n_samples=1, n_thermalization=50)[0]

print("\n" + "="*70)
print("Simulazione completata!")
# ==========================================
# Calcola CALORE SPECIFICO: C = dE/dT
# ==========================================
C_vals = []
for i in range(1, len(T_values) - 1):
    dE = E_vals[i+1] - E_vals[i-1]
    dT = T_values[i+1] - T_values[i-1]
    C = dE / dT
    C_vals.append(C)

# Padding per allineare con T_values
C_vals = [C_vals[0]] + C_vals + [C_vals[-1]]
# ===============================
# ===============================
# PLOT RISULTATI
# ===============================
fig, axes = plt.subplots(3, 3, figsize=(16, 14))

# Overlap m
axes[0, 0].plot(T_values, m_vals, 'b-o', lw=2, markersize=4)
axes[0, 0].set_xlabel("Temperatura T", fontsize=11)
axes[0, 0].set_ylabel("m (overlap)", fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title("Overlap vs T")

# q e q0
axes[0, 1].plot(T_values, q_vals, 'r-o', lw=2, markersize=4, label='q')
axes[0, 1].plot(T_values, q0_vals, 'g-s', lw=2, markersize=4, label='q₀')
axes[0, 1].set_xlabel("Temperatura T", fontsize=11)
axes[0, 1].set_ylabel("q, q₀", fontsize=11)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title("q e q₀ vs T")

# r e r0
axes[0, 2].plot(T_values, r_vals, 'm-o', lw=2, markersize=4, label='r')
axes[0, 2].plot(T_values, r0_vals, 'c-^', lw=2, markersize=4, label='r₀')
axes[0, 2].set_xlabel("Temperatura T", fontsize=11)
axes[0, 2].set_ylabel("r, r₀", fontsize=11)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_title("r e r₀ vs T")

# Energia interna
axes[1, 0].plot(T_values, E_vals, 'purple', marker='o', lw=2, markersize=4)
axes[1, 0].set_xlabel("Temperatura T", fontsize=11)
axes[1, 0].set_ylabel("E (energia interna)", fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title("Energia Interna vs T")

# Energia libera
axes[1, 1].plot(T_values, f_vals, 'orange', marker='o', lw=2, markersize=4)
axes[1, 1].set_xlabel("Temperatura T", fontsize=11)
axes[1, 1].set_ylabel("f (energia libera)", fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title("Energia libera vs T")

# Calore specifico
axes[1, 2].plot(T_values, C_vals, 'red', marker='o', lw=2, markersize=4)
axes[1, 2].set_xlabel("Temperatura T", fontsize=11)
axes[1, 2].set_ylabel("C = dE/dT", fontsize=11)
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_title("Calore Specifico (picco = transizione)")

# Q = 1 + 4(q - q0)
axes[2, 0].plot(T_values, 1 + 4*(np.array(q_vals) - np.array(q0_vals)), 'k-o', lw=2, markersize=4)
axes[2, 0].set_xlabel("Temperatura T", fontsize=11)
axes[2, 0].set_ylabel("Q = 1 + 4(q - q₀)", fontsize=11)
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_title("Parametro Q")

# Entropia S = (E - f) / T
S_vals = [(E_vals[i] - f_vals[i]) / T_values[i] if T_values[i] > 0.01 else 0 
          for i in range(len(T_values))]
axes[2, 1].plot(T_values, S_vals, 'brown', marker='o', lw=2, markersize=4)
axes[2, 1].set_xlabel("Temperatura T", fontsize=11)
axes[2, 1].set_ylabel("S = (E - f) / T", fontsize=11)
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].set_title("Entropia vs T")

# Differenze
axes[2, 2].plot(T_values, np.array(r_vals) - np.array(r0_vals), 'purple', marker='o', lw=2, markersize=4, label='r - r₀')
axes[2, 2].plot(T_values, np.array(q_vals) - np.array(q0_vals), 'brown', marker='s', lw=2, markersize=4, label='q - q₀')
axes[2, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2, 2].set_xlabel("Temperatura T", fontsize=11)
axes[2, 2].set_ylabel("Differenze", fontsize=11)
axes[2, 2].legend()
axes[2, 2].grid(True, alpha=0.3)
axes[2, 2].set_title("Differenze tra parametri")

plt.suptitle(f"Simulazione completa: N={N}, P={P}, α={alpha:.2f}, λ={lam:.2f}", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"complete_simulation_alpha={alpha:.2f}_lambda={lam:.2f}.png", dpi=150)
plt.show()

# ==========================================
# Trova temperatura critica dal picco di C
# ==========================================
C_array = np.array(C_vals)
T_c_idx = np.argmax(C_array)
T_c = T_values[T_c_idx]

print("\n" + "="*70)
print("STATISTICHE FINALI:")
print("="*70)
print(f"Temperatura critica (da picco C): T_c ≈ {T_c:.4f}")
print(f"Calore specifico massimo: C_max = {C_array[T_c_idx]:.4f}")
print(f"Overlap a T_c: m(T_c) = {m_vals[T_c_idx]:.4f}")
print(f"Energia a T_c: E(T_c) = {E_vals[T_c_idx]:.4f}")
print()
print(f"Overlap finale (T={T_values[-1]:.3f}): m = {m_vals[-1]:.4f}")
print(f"q - q₀ medio: {np.mean(np.array(q_vals) - np.array(q0_vals)):.6f}")
print(f"r - r₀ medio: {np.mean(np.array(r_vals) - np.array(r0_vals)):.6f}")
print(f"Energia libera finale: f = {f_vals[-1]:.4f}")
print(f"Energia finale: E = {E_vals[-1]:.4f}")
