# Boolean-Hopfield-Model

# Abstract
Attractor neural networks are systems composed of two-state discrete neurons which are able to store patterns and retrieve them starting from a noisy version.
Episodic memory is the collection of personal experiences that occurred at particular times and places.
The goal of this project is to study the temporal component, i.e. how memories are correlated in time, through a simplified model for episodic memory.
Experimental evidences show that enhanced synaptic strength is correlated with memory and that groups of neurons encode information in a way that minimizes the number of active neurons, reinforcing the hypotesis of the existence of engram cells, which are activated and modified by a learning experience and then reactivated by the presentation of a portion of the stimuli present at the learning experience.
We build a boolean version of the Hopfield Model (each neuron can take values V_i=0,1), where inputs encoding time are described as independent AR processes and lead to the formation of correlated patterns. These inputs mimick the slow random fluctuations of excitability of neurons which determine how engrams interact.
We derive the Boolean Hopfield Model free energy and selfconsistency equations with Guerra's interpolation method, then we explore the parameter space and we tune our internal field. Studying the critical temperature heatmap we find a critical value of temporal correlation lambda after which a new state begins to appear.
We find the new attractor is a mixed state composed of around 20 patterns near the one we're trying to retrieve. We study its properties and what happens to the order parameters, finding that the transition between single pattern retrieval and mixed state is a first order one, while from mixed state to paramagnetic we have a second order phase transition.
We try to retrieve the mixed state as a superposition of the patterns with a majority rule at fixed coding level. It shows a high overlap both when we try to retrieve the single pattern and when we place the system close to the mixture state.
Then, through a coarse-graining procedure, we build a system where the minima of the free energy are not the original patterns but the mixed state ones. The system shows a retrieval phase and a first order transition to a paramagnetic state just like the model with uncorrelated memories. 
Finally, we compare the overlapping ensemble at lambda>0.65, i.e. the number of neurons which activate together for two distinct patterns, with experimental data. Results match qualitatively.
We suggest the model is a good candidate for encoding episodic memories of subsequent events togheter, i.e. overlapping engrams. 
Moreover, we suggest an analogy with the notion of archetype in Artificial Neural Networks .

# Codes

You find the last version of every code which is part of the thesis project: 
---
1) Typical single attempt retrieval with the glauber dynamics
2) Montecarlo dynamics in order to check the order parameters state
3) Selfconsistency equations solved through the regularized fixed point algorithm (for lambda = 0)
4) 
5) 
