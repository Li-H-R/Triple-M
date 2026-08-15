# Theoretical Framework & Proofs

This directory contains the mathematical foundation for the representation learning model, focusing on **Semantic Enhancement** and **Detail Suppression**.

---

## 1. Theoretical Motivation

Let the input sample $\boldsymbol{X}$ be composed of two underlying generative factors:
* **Semantic Factor ($\mathfrak{s} \in \mathcal{S}$):** Information reflecting high-level turbine operations (e.g., operating modes).
* **Detail Factor ($\mathfrak{g} \in \mathcal{G}$):** Nuisance factors such as sensor-level variations, environmental fluctuations, or noise.

Let the encoder produce a normalized latent representation $\boldsymbol{z}$. 

### Mutual Information Formulations

1. **Semantic Mutual Information $I(\boldsymbol{z}; \mathfrak{s})$:** Measures how much semantic content $\boldsymbol{z}$ retains:
   $$I(\boldsymbol{z}; \mathfrak{s}) = \mathbb{E}_{p(\boldsymbol{z},\mathfrak{s})} \left[ \log \frac{p(\boldsymbol{z},\mathfrak{s})}{p(\boldsymbol{z})p(\mathfrak{s})} \right]$$

2. **Conditional Mutual Information $I(\boldsymbol{z}; \mathfrak{g} \mid \mathfrak{s})$:** Measures the residual detail information remaining in $\boldsymbol{z}$ after accounting for the semantics $\mathfrak{s}$.

> **Goal:** An optimal encoder maximizes $I(\boldsymbol{z}; \mathfrak{s})$ while minimizing $I(\boldsymbol{z}; \mathfrak{g} \mid \mathfrak{s})$.

---

## 2. Key Theoretical Results

### Theorem 1 (Semantic Enhancement and Detail Suppression)

Let $\boldsymbol{z} \in \mathbb{R}^d$ denote the normalized representation produced by the encoder ($\|\boldsymbol{z}\|_2 = 1$), and consider a contrastive learning framework with $N$ anchor samples.

Then, the conditional mutual information between $\boldsymbol{z}$ and the detail factor $\mathfrak{g}$ satisfies:

$$I(\boldsymbol{z};\mathfrak{g}\mid \mathfrak{s}) \le C - \log N + \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}$$

and the semantic mutual information satisfies:

$$I(\boldsymbol{z};\mathfrak{s}) \ge \log N - \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}$$

where $\mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}$ is the InfoNCE loss defined over $N$ samples:

$$\mathcal{L}_{\mathrm{NCE}_\mathfrak{s}} = -\mathbb{E}_i \left[ \log \frac{\exp(\mathrm{sim}(\mathfrak{s}_{i'},\boldsymbol{z}_i))}{\sum_{j=1}^{N}\exp(\mathrm{sim}(\mathfrak{s}_j,\boldsymbol{z}_i))} \right], \quad i' = 1, \dots, N$$

and $C$ represents the maximum differential entropy of a uniform distribution on the unit sphere $\mathbb{S}^{d-1}$:

$$C = \log \frac{2\pi^{d/2}}{\Gamma(d/2)}$$

---

### Robustness under Learned Semantics

If the learned semantic representation $s$ approximates the true underlying semantics $\mathfrak{s}$ such that $\| s - \mathfrak{s} \|_{2} \le \varepsilon$ for a sufficiently small $\varepsilon > 0$, then there exists a constant $K > 0$ satisfying:

$$\|\mathcal{L}_{\mathrm{NCE}_s} - \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}\|_2 \le K \| s - \mathfrak{s} \|_2$$

Consequently, the upper bound on detail mutual information becomes:

$$I(\boldsymbol{z};\mathfrak{g}\mid \mathfrak{s}) \le C - \log N + \mathcal{L}_{\mathrm{NCE}_s} + K\| s - \mathfrak{s} \|_{2}$$

---

## 3. Mathematical Proofs

<details>
<summary><b>Click to expand: Proof of Theorem 1</b></summary>

### Appendix I: Proof of Theorem 1

1. **InfoNCE Bound:** According to the InfoNCE mutual information lower bound, we have:

$$	I(\boldsymbol z;\mathfrak s_{i})
		$$

2. **Chain Rule Decomposition:**
   $$I(\boldsymbol{z}; (\mathfrak{s_i},\mathfrak{g})) = I(\boldsymbol{z}; \boldsymbol{X}) = I(\boldsymbol{z}; \mathfrak{s_i}) + I(\boldsymbol{z}; \mathfrak{g}\mid \mathfrak{s_i})$$

3. **Entropy Bound:** Since $I(\boldsymbol z; \boldsymbol X) \le H(\boldsymbol z)$:
   $$I(\boldsymbol z;\mathfrak g \mid \mathfrak s_i) \le H(\boldsymbol z) - I(\boldsymbol z;\mathfrak s_i) \le H(\boldsymbol z) - \log N + \mathcal L_{\mathrm{NCE}_{\mathfrak s_i}}$$

4. **Hyperspherical Bound:** Because $\|\boldsymbol z\|_2 = 1$, $\boldsymbol z$ lies on the unit sphere in $\mathbb{R}^d$. Its differential entropy is upper-bounded by the uniform distribution entropy:
   $$H(\boldsymbol z) \le \log\frac{2\pi^{d/2}}{\Gamma(d/2)} = C$$

5. **Final Aggregation:** Combining these steps yields:
   $$I(\boldsymbol z;\mathfrak g\mid\mathfrak s_i) \le C - \log N + \mathcal L_{\mathrm{NCE}_{\mathfrak s_i}}$$

Averaging over all anchors confirms that minimizing $\mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}$ simultaneously maximizes semantic retention and suppresses unwanted details. $\blacksquare$

</details>

<details>
<summary><b>Click to expand: Proof of the Lipschitz Inequality</b></summary>

### Appendix II: Proof of the Lipschitz Continuity

Let $\{s_i^+\}_{i=1}^N$ and $\{\mathfrak{s}_i^+\}_{i=1}^N$ be anchor sets with non-zero bounds ($\|s_i^+\|, \|\mathfrak{s}_i^+\|, \|{\boldsymbol x}_k\| \ge c > 0$), and temperature parameter $\tau > 0$.

1. **Cosine Similarity Lipschitz Property:** By Cauchy–Schwarz, there exists $C > 0$ such that:
   $$|\operatorname{sim}(s_i^+,{\boldsymbol x}_k) - \operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_k)| \le C \| s_i^+ - \mathfrak{s}_i^+ \|_2$$

2. **Loss Difference Bound:**
   $$|\mathcal{L}_{\mathrm{NCE}}(s_i^+) - \mathcal{L}_{\mathrm{NCE}}(\mathfrak{s}_i^+)| \le \frac{|\operatorname{sim}(s_i^+,{\boldsymbol x}_0) - \operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_0)|}{\tau} + \left| \log \sum_{k=0}^m e^{\frac{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)}{\tau}} - \log \sum_{k=0}^m e^{\frac{\operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_k)}{\tau}} \right|$$

3. **Log-Sum-Exp Lipschitz Continuity:**
   $$\left| \log S_{s_i^+} - \log S_{\mathfrak{s}_i^+} \right| \le \frac{\sum_{k=0}^m \left| e^{\frac{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)}{\tau}} - e^{\frac{\operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_k)}{\tau}} \right|}{\min(S_{s_i^+}, S_{\mathfrak{s}_i^+})}$$

4. **Mean Value Theorem Application:**
   $$\left| e^{\frac{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)}{\tau}} - e^{\frac{\operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_k)}{\tau}} \right| \le \frac{C}{\tau} e^{1/\tau} \| s_i^+ - \mathfrak{s}_i^+ \|_2$$

5. **Final Constant:** Combining terms yields the Lipschitz constant $K = \frac{C}{\tau}(1 + e^{2/\tau})$:
   $$\| \mathcal{L}_{\mathrm{NCE}_s} - \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}} \|_2 \le K \| s^+ - \mathbf{\mathfrak{s}}^+ \|_2$$
$\blacksquare$

</details>
## Reference

[1]contrastive predictive coding."arXiv, 2018 Representation Jearning withk
