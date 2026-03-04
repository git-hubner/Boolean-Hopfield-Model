# A simplified model for episodic memory

We build a network of $N$ binary neurons whose state is $V_i = 0,1$, and connectivity matrix $J_{ij}$.

The patterns experienced by the network and stored in the connectivity matrix are defined by:

$$
\eta_i^t = \Theta(I_{i,\text{episode}}^t + I_{i,\text{time}}^t - T)
$$

where:
- $I_{i,\text{episode}}^t$ are i.i.d. random inputs encoding the episode occurring at time $t$, described by a Gaussian random variable with zero mean and variance $\sigma_e^2$;
- $I_{i,\text{time}}$ is an input encoding time, described as independent AR processes, where at each time $t$:

$$
I_{i,\text{time}}^t = \lambda I_{i,\text{time}}^{t-1} + \sigma_t \sqrt{1-\lambda^2} z_i^t
$$

with $z_i^t$ i.i.d. random variables. Consequently, $I_{i,\text{time}}^t$ are Gaussian with variance $\sigma_t^2$, and $T$ is a threshold that can be set so that for any given episode, a fraction $f$ of neurons are active.

The biological motivation for the time encoding input is the following: slow random endogenous fluctuations of excitability of neurons determine how engrams interact, and we simulate these fluctuations with an autoregressive process. Since the sum of the currents is Gaussian, $T$ is related to $f$ by:

$$
f = \int_T^{\infty} \frac{1}{\sqrt{2\pi(\sigma_e^2+\sigma_t^2)}} \exp\left(-\frac{z^2}{2(\sigma_e^2+\sigma_t^2)}\right) dz = \frac{1}{2}\text{erfc}\left(\frac{T}{\sqrt{2}}\right)
$$

Without loss of generality, we can set $\sigma_e^2 + \sigma_t^2 = 1$.

## Learning

These "episodic" patterns are stored in the connectivity matrix using a standard Hebbian rule:

$$
J_{ij} = \frac{1}{N f (1-f)} \sum_{\mu=1}^P (\eta_i^\mu - f)(\eta_j^\mu - f) 
$$

i.e., the covariance matrix introduced by Tsodyks and Feigel'man [1988].

The network is described by four parameters:
- f: coding level;
- one of the two variances, $\sigma_e$ or $\sigma_t$ (since the sum of their squares is equal to 1);
- $\lambda$: the correlation between subsequent time inputs;
- P: the number of stored episodes.

The correlation parameter $\lambda = 1 - \frac{a}{N}$ where:
- if $a \sim O(1)$, we have a correlation time of order $N$;
- if $a \sim O(N)$, we have a correlation time of order 1.

## Retrieval of episodes

To check whether episodes are retrieved, we start with an initial condition correlated with one of the stored patterns and run network dynamics. We use Glauber dynamics (see Appendix \ref{chap:glauber}), where each neuron has a probability of being active:

$$
\text{Prob}(V_i \rightarrow 1) = \frac{1}{1 + e^{-\beta(h_i - \theta)}}
$$

with $h_i = \sum_{j \neq i}^N J_{ij} V_j$ being the internal field acting on the $i$-th spin.

The Hamiltonian in our case is:

$$
H(\mathbf{V}) = -\frac{1}{2} \sum_{ij} J_{ij} V_i V_j + \theta \sum_i V_i 
$$

The Mattis magnetization with the $\mu$-th pattern for the Boolean Hopfield Model with synaptic matrix \eqref{eq:Jij} is:

$$
m^\mu = \frac{1}{f(1-f)N} \sum_{i=1}^N V_i (\eta_i^\mu - f)
$$
