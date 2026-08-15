# Theoretical Foundation: Semantic Enhancement & Detail Suppression

This document presents the theoretical motivation and proofs for the contrastive learning framework applied to wind turbine operational representations. The framework explicitly maximizes semantic mutual information while suppressing unnecessary detail noise (e.g., sensor-level variations).

---

## 1. Theoretical Motivation

Let the input sample $\boldsymbol{X}$ be governed by two underlying generative factors:
* **Semantic factor ($\mathfrak{s} \in \mathcal{S}$):** Reflects high-level turbine operations (e.g., operating modes).
* **Detail factor ($\mathfrak{g} \in \mathcal{G}$):** Reflects fine-grained noise or sensor-level variations.

Let the encoder yield a normalized latent representation $\boldsymbol{z} \in \mathbb{R}^d$ with $\|\boldsymbol{z}\|_2 = 1$.

### Mutual Information Formulations

1. **Semantic Mutual Information:** Measures the amount of semantic content retained by $\boldsymbol{z}$:
   $$I(\boldsymbol{z}; \mathfrak{s}) = \mathbb{E}_{p(\boldsymbol{z},\mathfrak{s})} \left[ \log \frac{p(\boldsymbol{z},\mathfrak{s})}{p(\boldsymbol{z})p(\mathfrak{s})} \right]$$

2. **Conditional Mutual Information:** Measures detail noise retained in $\boldsymbol{z}$ after accounting for $\mathfrak{s}$:
   $$I(\boldsymbol{z}; \mathfrak{g} \mid \mathfrak{s})$$

> **Goal:** An optimal encoder should **maximize** $I(\boldsymbol{z};\mathfrak{s})$ and **minimize** $I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s})$.

---

## 2. Main Theoretical Result

### Theorem 1 (Semantic Enhancement and Detail Suppression)

> **Theorem Statement**  
> Let $\boldsymbol{z} \in \mathbb{R}^d$ be a normalized representation ($\|\boldsymbol{z}\|_2 = 1$) under a contrastive learning setup with $N$ anchor samples. The conditional mutual information between $\boldsymbol{z}$ and detail factor $\mathfrak{g}$ satisfies:
>
> $$I(\boldsymbol{z};\mathfrak{g}\mid \mathfrak{s}) \le C - \log N + \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}$$
>
> and the semantic mutual information is lower-bounded by:
>
> $$I(\boldsymbol{z};\mathfrak{s}) \ge \log N - \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}$$
>
> where $\mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}$ is the InfoNCE-type contrastive loss under ideal semantics:
>
> $$\mathcal{L}_{\mathrm{NCE}_\mathfrak{s}} = -\mathbb{E}_i \left[ \log \frac{\exp(\mathrm{sim}(\mathfrak{s}_{i'},\boldsymbol{z}_i))}{\sum_{j=1}^{N}\exp(\mathrm{sim}(\mathfrak{s}_j,\boldsymbol{z}_i))} \right], \quad i' \in \{1, \dots, N\}$$
>
> and the normalization constant $C$ (surface area term of the unit hypersphere) is defined as:
>
> $$C = \log \frac{2\pi^{d/2}}{\Gamma(d/2)}$$

Furthermore, if the learned semantic representation $s$ approximates the true semantics $\mathfrak{s}$ such that $\| s - \mathfrak{s} \|_{2} \le \varepsilon$ for a small $\varepsilon > 0$, there exists a constant $K > 0$ such that:

$$\|\mathcal{L}_{\mathrm{NCE}_s} - \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}\|_2 \le K \| s - \mathfrak{s} \|_2$$

This yields the practical operational bound:

$$I(\boldsymbol{z};\mathfrak{g}\mid \mathfrak{s}) \le C - \log N + \mathcal{L}_{\mathrm{NCE}_s} \pm K\| s - \mathfrak{s} \|_{2}$$

---

# 3. Supplementary Proofs

## Appendix I: Proof of Theorem 1

1. **InfoNCE Lower Bound:** By definition of the InfoNCE lower bound:
   $$I(\boldsymbol{z}; \mathfrak{s}_i) \ge \log N - \mathcal{L}_{\mathrm{NCE}_{\mathfrak{s}_i}}$$

2. **Chain Rule Decomposition:**
   $$I(\boldsymbol{z}; (\mathfrak{s}_i, \mathfrak{g})) = I(\boldsymbol{z}; \boldsymbol{X}) = I(\boldsymbol{z}; \mathfrak{s}_i) + I(\boldsymbol{z}; \mathfrak{g} \mid \mathfrak{s}_i)$$

3. **Entropy Bound:** Since $I(\boldsymbol{z}; \boldsymbol{X}) \le H(\boldsymbol{z})$, we have:
   $$I(\boldsymbol{z}; \mathfrak{g} \mid \mathfrak{s}_i) \le H(\boldsymbol{z}) - I(\boldsymbol{z}; \mathfrak{s}_i) \le H(\boldsymbol{z}) - \log N + \mathcal{L}_{\mathrm{NCE}_{\mathfrak{s}_i}}$$

4. **Hypersphere Maximum Entropy:** Because $\|\boldsymbol{z}\|_2 = 1$, $\boldsymbol{z}$ lies on the unit sphere in $\mathbb{R}^d$, bounding its differential entropy by $C$:
   $$H(\boldsymbol{z}) \le \log \frac{2\pi^{d/2}}{\Gamma(d/2)} = C$$

Combining these steps proves the bound for all anchors $i = 1, \dots, N$:

$$I(\boldsymbol{z}; \mathfrak{g} \mid \mathfrak{s}) \le C - \log N + \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}} \quad \blacksquare$$

---

## Appendix II: Proof of the Lipschitz Inequality

Let $\{s_i^+\}_{i=1}^N$ and $\{\mathfrak{s}_i^+\}_{i=1}^N$ be anchor sets with bounded norms $\|s_i^+\|, \|\mathfrak{s}_i^+\|, \|\boldsymbol{x}_k\| \ge c > 0$.

1. **Cosine Similarity Lipschitz Continuity:** By Cauchy–Schwarz, there exists $C > 0$ such that:
   $$|\mathrm{sim}(s_i^+, \boldsymbol{x}_k) - \mathrm{sim}(\mathfrak{s}_i^+, \boldsymbol{x}_k)| \le C \| s_i^+ - \mathfrak{s}_i^+ \|_2$$

2. **Log-Sum-Exp Bound:** Using the Lipschitz property of the $\log\sum\exp$ function:
   $$\left| \mathcal{L}_{\mathrm{NCE}}(s_i^+) - \mathcal{L}_{\mathrm{NCE}}(\mathfrak{s}_i^+) \right| \le \frac{|\Delta \mathrm{sim}_0|}{\tau} + \frac{\sum_{k=0}^m \left| e^{\mathrm{sim}(s_i^+, \boldsymbol{x}_k)/\tau} - e^{\mathrm{sim}(\mathfrak{s}_i^+, \boldsymbol{x}_k)/\tau} \right|}{\min(S_{s_i^+}, S_{\mathfrak{s}_i^+})}$$

3. **Applying Mean Value Theorem:**
   $$\left| e^{\mathrm{sim}(s_i^+, \boldsymbol{x}_k)/\tau} - e^{\mathrm{sim}(\mathfrak{s}_i^+, \boldsymbol{x}_k)/\tau} \right| \le \frac{C}{\tau} e^{1/\tau} \| s_i^+ - \mathfrak{s}_i^+ \|_2$$

4. **Final Bound Aggregation:** Combining terms yields the constant $K = \frac{C}{\tau} \left( 1 + e^{2/\tau} \right)$, giving the final aggregated inequality:

$$\big\| \mathcal{L}_{\mathrm{NCE}_s} - \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}} \big\|_2 \le K \big\| s^+ - \mathbf{\mathfrak{s}}^+ \big\|_2 \quad \blacksquare$$
