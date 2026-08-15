# Theoretical Motivation and Proofs

## Theoretical Motivation

Let the input $\boldsymbol{X}$ contain two underlying generative factors:
* **Semantic factor ($\boldsymbol{s} \in \mathcal{S}$):** Reflects turbine operation (e.g., operating mode).
* **Detail factor ($\boldsymbol{g} \in \mathcal{G}$):** Represents sensor-level variations or noise.

Let the encoder produce a normalized latent representation $\boldsymbol{z}$. 

### Mutual Information Definitions

1. **Semantic Mutual Information:** Measures how much semantic content $\boldsymbol{z}$ retains:

$$
I(\boldsymbol{z}; \boldsymbol{s}) = \mathbb{E}_{p(\boldsymbol{z},\boldsymbol{s})} \left[ \log \frac{p(\boldsymbol{z},\boldsymbol{s})}{p(\boldsymbol{z})p(\boldsymbol{s})} \right]
$$

2. **Conditional Mutual Information:** Measures the amount of detail information contained in $\boldsymbol{z}$ after accounting for $\boldsymbol{s}$:

$$
I(\boldsymbol{z}; \boldsymbol{g}\mid \boldsymbol{s})
$$

A desirable encoder should maximize $I(\boldsymbol{z};\boldsymbol{s})$ and minimize $I(\boldsymbol{z};\boldsymbol{g}\mid\boldsymbol{s})$ to alleviate representation collapse / spurious learning.

---

### Theorem 1 (Semantic Enhancement and Detail Suppression)

> **Theorem 1.** Let $\boldsymbol{z} \in \mathbb{R}^d$ denote the normalized representation produced by the encoder, i.e., $\|\boldsymbol{z}\|_2 = 1$, and consider a contrastive learning framework with $N$ anchor samples.
>
> Then the conditional mutual information between $\boldsymbol{z}$ and the detail factor $\boldsymbol{g}$ satisfies:
>
> $$
> I(\boldsymbol{z};\boldsymbol{g}\mid \boldsymbol{s}) \le C - \log N + \mathcal{L}_{\mathrm{NCE}_{\boldsymbol{s}}}
> $$
>
> and
>
> $$
> I(\boldsymbol{z};\boldsymbol{s}) \ge \log N - \mathcal{L}_{\mathrm{NCE}_{\boldsymbol{s}}}
> $$
>
> where $\mathcal{L}_{\mathrm{NCE}_{\boldsymbol{s}}}$ is an InfoNCE-type contrastive loss induced by a mode-matching mechanism under ideal wind turbine semantics ($i' \in \{1, \dots, N\}$):
>
> $$
> \mathcal{L}_{\mathrm{NCE}_{\boldsymbol{s}}} = -\mathbb{E}_i \left[ \log \frac{\exp(\mathrm{sim}(\boldsymbol{s}_{i'},\boldsymbol{z}_i))}{\sum_{j=1}^{N}\exp(\mathrm{sim}(\boldsymbol{s}_j,\boldsymbol{z}_i))} \right]
> $$
>
> and the normalization constant $C$ is given by:
>
> $$
> C = \log \frac{2\pi^{d/2}}{\Gamma(d/2)}
> $$
>
> with $\Gamma(\cdot)$ denoting the Gamma function.

*(The proof is provided in Appendix I below.)*

Furthermore, if the learned semantic representation $s$ satisfies $\| s - \boldsymbol{s} \|_{2} \le \varepsilon$ for a sufficiently small $\varepsilon > 0$, then there exists a constant $K > 0$ such that:

$$
\|\mathcal{L}_{\mathrm{NCE}_s} - \mathcal{L}_{\mathrm{NCE}_{\boldsymbol{s}}}\|_2 \le K \| s - \boldsymbol{s} \|_2
$$

and

$$
I(\boldsymbol{z};\boldsymbol{g}\mid \boldsymbol{s}) \le C - \log N + \mathcal{L}_{\mathrm{NCE}_s} \pm K\| s - \boldsymbol{s} \|_{2}
$$

Thus, minimizing the loss $\mathcal{L}_{\mathrm{NCE}_s}$ associated with the true semantics $\boldsymbol{s}$ increases the mutual information $I(\boldsymbol{z};\boldsymbol{s})$ and tightens the upper bound of $I(\boldsymbol{z};\boldsymbol{g} \mid \boldsymbol{s})$, ensuring that the representation $\boldsymbol{z}$ increasingly reflects semantic structure.

---

## Appendix I: Proof of Theorem 1

1. **InfoNCE Lower Bound:**  
   According to the InfoNCE bound, the mutual information between $\boldsymbol{z}$ and the semantic factor $\boldsymbol{s}_{i}$ satisfies:
   $$
   I(\boldsymbol{z};\boldsymbol{s}_{i}) \ge \log N - \mathcal{L}_{\mathrm{NCE}_{\boldsymbol{s}_i}}
   $$

2. **Chain Rule of Mutual Information:**  
   $$
   I(\boldsymbol{z}; (\boldsymbol{s}_i,\boldsymbol{g})) = I(\boldsymbol{z}; \boldsymbol{X}) = I(\boldsymbol{z}; \boldsymbol{s}_i) + I(\boldsymbol{z}; \boldsymbol{g}\mid \boldsymbol{s}_i)
   $$

3. **Bounding Detail Mutual Information:**  
   Since $I(\boldsymbol{z};\boldsymbol{X}) \le H(\boldsymbol{z})$, we obtain:
   $$
   I(\boldsymbol{z};\boldsymbol{g}\mid\boldsymbol{s}_i) \le H(\boldsymbol{z}) - I(\boldsymbol{z};\boldsymbol{s}_i)
   $$

4. **Substituting the InfoNCE Bound:**  
   $$
   I(\boldsymbol{z};\boldsymbol{g}\mid\boldsymbol{s}_i) \le H(\boldsymbol{z}) - \log N + \mathcal{L}_{\mathrm{NCE}_{\boldsymbol{s}_i}}
   $$

5. **Spherical Entropy Bound:**  
   Since $\|\boldsymbol{z}\|_2 = 1$, $\boldsymbol{z}$ lies on the unit hypersphere in $\mathbb{R}^d$. Its differential entropy is bounded by the uniform distribution on the sphere:
   $$
   H(\boldsymbol{z}) \le \log \frac{2\pi^{d/2}}{\Gamma(d/2)} = C
   $$

6. **Conclusion:**  
   Combining the bounds yields:
   $$
   I(\boldsymbol{z};\boldsymbol{g}\mid\boldsymbol{s}_i) \le C - \log N + \mathcal{L}_{\mathrm{NCE}_{\boldsymbol{s}_i}}
   $$

   Since each $\boldsymbol{s}_i$ represents semantic information within the wind turbine semantic space, aggregating across samples gives:
   $$
   I(\boldsymbol{z};\boldsymbol{g}\mid \boldsymbol{s}) \le C - \log N + \mathcal{L}_{\mathrm{NCE}_{\boldsymbol{s}}}
   $$

   Similarly, the lower bound $I(\boldsymbol{z};\boldsymbol{s}) \ge \log N - \mathcal{L}_{\mathrm{NCE}_{\boldsymbol{s}}}$ follows directly from the InfoNCE inequality. $\blacksquare$

---

## Appendix II: Proof of the Lipschitz Inequality

Let $\{s_i^+\}_{i=1}^N$ and $\{\boldsymbol{s}_i^+\}_{i=1}^N$ be two sets of anchor vectors from different semantic Hilbert spaces, with corresponding positive and negative samples $\{{\boldsymbol x}_k\}_{k=0}^m$. 

Assume all vectors have norms bounded away from zero:
$$
\|s_i^+\|, \|\boldsymbol{s}_i^+\|, \|{\boldsymbol x}_k\| \ge c > 0, \quad \forall i,k
$$

Let $\tau > 0$ be the temperature parameter. The NCE loss for anchor $s_i^+$ is defined as:
$$
\mathcal{L}_{\mathrm{NCE}}(s_i^+) = -\frac{\operatorname{sim}(s_i^+,{\boldsymbol x}_{k'})}{\tau} + \log\left(\sum_{k=0}^{m} \exp\left(\frac{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)}{\tau}\right)\right)
$$

where $\operatorname{sim}(s_i^+,{\boldsymbol x}) = \frac{s_i^+ \cdot {\boldsymbol x}}{\|s_i^+\|\|{\boldsymbol x}\|}$ is the cosine similarity, and ${\boldsymbol x}_{k'}$ is the positive sample for $s_i^+$.

### Step-by-Step Derivation

1. **Cosine Similarity Lipschitz Continuity:**  
   Based on the Cauchy–Schwarz inequality and norm properties, there exists a constant $C > 0$ such that:
   $$
   |\operatorname{sim}(s_i^+,{\boldsymbol x}_k) - \operatorname{sim}(\boldsymbol{s}_i^+,{\boldsymbol x}_k)| \le C \| s_i^+ - \boldsymbol{s}_i^+ \|_2, \quad \forall i,k
   $$

2. **Bounding Loss Difference:**  
   The difference in NCE loss between anchor representations $s_i^+$ and $\boldsymbol{s}_i^+$ is bounded by:
   $$
   \big|\mathcal{L}_{\mathrm{NCE}}(s_i^+) - \mathcal{L}_{\mathrm{NCE}}(\boldsymbol{s}_i^+)\big| \le \frac{|\operatorname{sim}(s_i^+,{\boldsymbol x}_0) - \operatorname{sim}(\boldsymbol{s}_i^+,{\boldsymbol x}_0)|}{\tau} + \left| \log \sum_{k=0}^m e^{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)/\tau} - \log \sum_{k=0}^m e^{\operatorname{sim}(\boldsymbol{s}_i^+,{\boldsymbol x}_k)/\tau} \right|
   $$

3. **Log-Sum-Exp Lipschitz Property:**  
   Applying the mean value theorem / Lipschitz property to the $\log\sum\exp$ term:
   $$
   \left| \log \sum_{k=0}^m e^{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)/\tau} - \log \sum_{k=0}^m e^{\operatorname{sim}(\boldsymbol{s}_i^+,{\boldsymbol x}_k)/\tau} \right| \le \frac{\sum_{k=0}^m \big| e^{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)/\tau} - e^{\operatorname{sim}(\boldsymbol{s}_i^+,{\boldsymbol x}_k)/\tau} \big|}{\min(S_{s_i^+}, S_{\boldsymbol{s}_i^+})}
   $$

   where:
   $$
   S_{s_i^+} = \sum_{k=0}^m e^{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)/\tau}, \quad S_{\boldsymbol{s}_i^+} = \sum_{k=0}^m e^{\operatorname{sim}(\boldsymbol{s}_i^+,{\boldsymbol x}_k)/\tau}
   $$

4. **Exponential Bounding:**  
   Applying the Lagrange Mean Value Theorem to the exponential terms yields:
   $$
   \big| e^{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)/\tau} - e^{\operatorname{sim}(\boldsymbol{s}_i^+,{\boldsymbol x}_k)/\tau} \big| \le \frac{C}{\tau} e^{1/\tau} \| s_i^+ - \boldsymbol{s}_i^+ \|_2
   $$

5. **Single Anchor Lipschitz Constant:**  
   Combining the bounds gives the inequality for the $i$-th anchor:
   $$
   \big|\mathcal{L}_{\mathrm{NCE}}(s_i^+) - \mathcal{L}_{\mathrm{NCE}}(\boldsymbol{s}_i^+)\big| \le K \| s_i^+ - \boldsymbol{s}_i^+ \|_2
   $$

   where:
   $$
   K = \frac{C}{\tau} \left( 1 + e^{2/\tau} \right)
   $$

6. **Global Bounds Aggregation:**  
   Aggregating across all anchors $i = 1, \dots, N$:
   $$
   \big\| \mathcal{L}_{\mathrm{NCE}_s} - \mathcal{L}_{\mathrm{NCE}_{\boldsymbol{s}}} \big\|_2 \le K \big\| s^+ - \boldsymbol{s}^+ \big\|_2
   $$

   where $s^+ = [s_1^+,\dots,s_N^+]$ and $\boldsymbol{s}^+ = [\boldsymbol{s}_1^+,\dots,\boldsymbol{s}_N^+]$. $\blacksquare$
