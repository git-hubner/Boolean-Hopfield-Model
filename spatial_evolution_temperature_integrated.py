import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime
import itertools

# ==================== PARAMETRI GLOABLI ====================
N = 10000
f = 0.1
s_e = 0.7
s_t = np.sqrt(1 - s_e**2)
T_threshold = norm.ppf(1 - f)

# Simulation temperature grid
T_range = np.linspace(0.001, 0.5, 50)

# targets to test
targets = list(range(97, 104))   # 97..103 inclusive

# Experiment settings to sweep
lambda_list = [0.6, 0.7, 0.75, 0.8]
alpha_list = [0.05, 0.1, 0.2]

# Monte Carlo parameters
MC_steps = 30
init_flip_fraction = 1/30   # same as original

# Output timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------- utilitarie ----------------

def prob(h, tht, beta):
    with np.errstate(over='ignore'):
        return 1.0/(1.0 + np.exp(-(h - tht) * beta))


def magn(Z, eta_p, f):
    return np.dot(eta_p - f, Z) / ((1 - f) * np.sum(eta_p))


def generate_patterns(P, N, lam, s_e, s_t, T_threshold):
    eta = np.zeros((P, N))
    I_ep = np.random.normal(0, s_e, size=(P, N))
    I_t = np.zeros((P, N))
    if P <= 0:
        return eta
    # initialize temporal correlated inputs
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


def run_retrieval_single(J, eta_pattern, theta, beta, steps=MC_steps):
    """Run Glauber-like dynamics starting from a noisy copy of eta_pattern.
    Returns final W (0/1 array).
    """
    N = len(eta_pattern)
    W = eta_pattern.copy()
    flip_idx = np.random.choice(N, size=int(N * init_flip_fraction), replace=False)
    W[flip_idx] = 1 - W[flip_idx]

    for _ in range(steps):
        for i in range(N):
            h = np.dot(J[i], W)
            phi = prob(h, theta, beta)
            W[i] = 1 if np.random.rand() < phi else 0
    return W


# ---------------- main experiment ----------------

all_results = {}

for lam in lambda_list:
    for alpha in alpha_list:
        print(f"Starting run: lambda={lam}, alpha={alpha}")
        P = int(alpha * N)
        if P < max(targets) + 1:
            # ensure we have enough patterns to index targets
            P_needed = max(targets) + 1
            P = P_needed
            alpha_eff = P / N
            print(f"  adjusted P to {P} (alpha_eff={alpha_eff:.4f}) to include requested targets")
        # generate patterns and J
        eta = generate_patterns(P, N, lam, s_e, s_t, T_threshold)
        J = build_J_matrix(eta, f, N)

        # prepare storage for this (lam,alpha)
        key = (lam, alpha)
        all_results[key] = {
            'T': list(T_range),
            'm': {mu: [] for mu in targets},     # m[mu] is list over T
            'q0': [],
            'pair_overlap': {pair: [] for pair in itertools.combinations(targets, 2)}
        }

        # For each temperature run retrieval for each target and compute overlaps
        for t_idx, T in enumerate(T_range):
            beta = 1.0 / T
            # store Ws for this T
            W_dict = {}
            for mu in targets:
                eta_mu = eta[mu]
                Wmu = run_retrieval_single(J, eta_mu, theta, beta, steps=MC_steps)
                W_dict[mu] = Wmu
                # compute overlap m with its native pattern
                mval = magn(Wmu, eta_mu, f)
                all_results[key]['m'][mu].append(mval)
            # global activity (we take activity of target 100's state as representative)
            all_results[key]['q0'].append(np.sum(W_dict[targets[0]]) / N)

            # compute pairwise simultaneous activation between Ws
            for (mu, nu) in itertools.combinations(targets, 2):
                Wmu = W_dict[mu]
                Wnu = W_dict[nu]
                common = np.sum((Wmu == 1) & (Wnu == 1)) / N
                all_results[key]['pair_overlap'][(mu, nu)].append(common)

            if (t_idx + 1) % 10 == 0:
                print(f"  lambda={lam} alpha={alpha} T-index {t_idx+1}/{len(T_range)}")

        # save a compact text summary for this run
        outname = f"overlap_lambda{lam:.2f}_alpha{alpha:.3f}_{timestamp}.npz"
        np.savez_compressed(outname, **all_results[key])
        print(f"Saved results to {outname}")

print("All experiments completed.")

# ---------------- optional plotting for one example ----------------
# plot pair overlap heatmap for a chosen (lam,alpha)
example_key = (0.75, 0.1) if (0.75, 0.1) in all_results else next(iter(all_results.keys()))
res = all_results[example_key]

# create matrix pairs x T
pairs = list(itertools.combinations(targets, 2))
pair_matrix = np.array([res['pair_overlap'][p] for p in pairs])  # shape (21, nT)

plt.figure(figsize=(10, 6))
plt.imshow(pair_matrix, aspect='auto', origin='lower', cmap='viridis',
           extent=[res['T'][0], res['T'][-1], 0, len(pairs)])
plt.yticks(np.arange(len(pairs)) + 0.5, [f"{a}-{b}" for (a,b) in pairs])
plt.colorbar(label='fraction simultaneous active')
plt.xlabel('Temperature')
plt.title(f'Pairwise simultaneous activation, lambda={example_key[0]}, alpha={example_key[1]}')
plt.tight_layout()
plt.show()

