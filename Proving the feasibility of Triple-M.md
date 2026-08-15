Here is the Markdown version of your content, formatted specifically for a GitHub `README.md`. It includes proper LaTeX rendering for math (using standard Markdown math syntax) and clear section headers.
***
# Theoretical Motivation
Let the input $\boldsymbol{X}$ contain two underlying generative factors:
*   A **semantic factor** reflecting the turbine operation $\mathfrak{s}\in\mathcal{S}$ (e.g., operating mode).
*   A **detail factor** $\mathfrak{g}\in\mathcal{G}$ (e.g., sensor-level variations or noise).
Let the encoder produce a normalized latent representation $\boldsymbol{z}$. The **semantic mutual information** is defined as:
$$
I(\boldsymbol{z}; \mathfrak{s}) = \mathbb{E}_{p(\boldsymbol{z},\mathfrak{s})} \Bigg[ \log \frac{p(\boldsymbol{z},\mathfrak{s})}{p(\boldsymbol{z})p(\mathfrak{s})} \Bigg],
$$
measuring how much semantic content $\boldsymbol{z}$ retains.
Similarly, the **conditional mutual information**:
$$
I(\boldsymbol{z}; \mathfrak{g}\mid \mathfrak{s})
$$
measures the amount of detail information contained in $\boldsymbol{z}$ after accounting for $\mathfrak{s}$.
A desirable encoder should therefore **maximize** $I(\boldsymbol{z};\mathfrak{s})$ and **minimize** $I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s})$ to alleviate the Semantic Loss (SL).
---
## Theorem 1: Semantic Enhancement and Detail Suppression
*Let $\boldsymbol{z} \in \mathbb{R}^d$ denote the normalized representation produced by the encoder, i.e., $\|\boldsymbol{z}\|_2 = 1$, and consider a contrastive learning framework with $N$ anchor samples.*
*Then the conditional mutual information between $\boldsymbol{z}$ and the detail factor $\mathfrak{g}$ satisfies*
$$
I(\boldsymbol{z};\mathfrak{g}\mid \mathfrak{s}) \;\le\; C - \log N + \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}},
$$
*and*
$$
I(\boldsymbol{z};\mathfrak{s}) \ge \log N - \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}},
$$
*where*
$$
\mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}= -\mathbb{E}_i \left[ \log \frac{\exp(\mathrm{sim}(\mathfrak{s}_{i'},\boldsymbol{z}_i))} {\sum_{j=1}^{N}\exp(\mathrm{sim}(\mathfrak{s}_j,\boldsymbol{z}_i))} \right], \quad i' = 1, \dots, N
$$
*is an InfoNCE-type contrastive loss induced by a mode-matching mechanism under ideal wind turbine semantics.*
$$
C=\log\frac{2\pi^{d/2}}{\Gamma(d/2)},
$$
*and $\Gamma(\cdot)$ denotes the Gamma function.*
### Robustness Analysis
Furthermore, if the learned semantic representation $s$ satisfies
$$
\| s - \mathfrak{s} \|_{2} \le \varepsilon,
$$
for a sufficiently small $\varepsilon > 0$, then there exists a constant $K > 0$ such that:
$$
\|\mathcal{L}_{\mathrm{NCE}_s} - \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}\|_2 \le K \| s - \mathfrak{s} \|_2
$$
and
$$
I(\boldsymbol{z};\mathfrak{g}\mid \mathfrak{s}) \;\le\; C - \log N + \mathcal{L}_{\mathrm{NCE}_s} \pm K\| s - \mathfrak{s} \|_{2}.
$$
Thus, minimizing the loss $\mathcal{L}_{\mathrm{NCE}_s}$ associated with the true semantics $\mathfrak{s}$ can increase the mutual information $I(\boldsymbol{z};\mathfrak{s})$ and tighten the upper bound of $I(\boldsymbol{z};\mathfrak{g} \mid \mathfrak{s})$, thereby ensuring that the representation $\boldsymbol{z}$ increasingly reflects semantic structure.
---
## Appendix I: Proof of Theorem 1
According to the InfoNCE loss definition and the InfoNCE bound[^1], the mutual information between $\boldsymbol z$ and the semantic factor $\mathfrak{s}_{i}$ satisfies:
$$
I(\boldsymbol z;\mathfrak s_{i}) \ge \log N-\mathcal L_{\mathrm{NCE}_{{\mathfrak s}_{i}}}.
$$
By the chain rule of mutual information:
$$
I(\boldsymbol{z}; (\mathfrak{s_i},\mathfrak{g})) = I(\boldsymbol{z}; \boldsymbol{X}) = I(\boldsymbol{z}; \mathfrak{s_i}) + I(\boldsymbol{z}; \mathfrak{g}\mid \mathfrak{s_i}).
$$
Since $I(\boldsymbol z;\boldsymbol X)\le H(\boldsymbol z)$, we obtain:
$$
I(\boldsymbol z;\mathfrak g|\mathfrak s_i) \le H(\boldsymbol z)-I(\boldsymbol z;\mathfrak s_i).
$$
Substituting the InfoNCE bound yields:
$$
I(\boldsymbol z;\mathfrak g|\mathfrak s_i) \le H(\boldsymbol z)-\log N+\mathcal L_{\mathrm{NCE}_{\mathfrak s_i}}.
$$
Since $\|\boldsymbol z\|_2=1$, $\boldsymbol z$ lies on the unit hypersphere in $\mathbb{R}^d$, and its entropy satisfies:
$$
H(\boldsymbol z)\le \log\frac{2\pi^{d/2}}{\Gamma(d/2)}=C.
$$
Therefore,
$$
I(\boldsymbol z;\mathfrak g|\mathfrak s_i) \le C-\log N+\mathcal L_{\mathrm{NCE}_{\mathfrak s_i}}.
$$
The above equation holds for all $ i = 1, \dots, N $. Since each $ \mathfrak{s}_i$ represents semantic information within the wind turbine semantic space, increasing the mutual information implies that more semantic information is captured in the representation $\boldsymbol z$. Hence, we have:
$$
I(\boldsymbol{z};\mathfrak{g}\mid \mathfrak{s}) \le C - \log N + \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}}.
$$
Similarly, the lower bound of $I(\boldsymbol z;\mathfrak s)$ follows directly from the InfoNCE inequality. $\blacksquare$
---
## Appendix II: Proof of the Lipschitz Inequality
Let $\{s_i^+\}_{i=1}^N$ and $\{\mathfrak{s}_i^+\}_{i=1}^N$ be two sets of anchor vectors from different semantic Hilbert spaces, with corresponding positive and negative samples $\{{\boldsymbol x}_k\}_{k=0}^m$. Assume all vectors have norms bounded away from zero:
$$
\|s_i^+\|, \|\mathfrak{s}_i^+\|, \|{\boldsymbol x}_k\| \ge c > 0, \quad \forall i,k,
$$
and let $\tau>0$ be the temperature parameter.
The NCE loss for anchor $s_i^+$ is defined as:
$$
\mathcal{L}_{\mathrm{NCE}}(s_i^+) = -\frac{\operatorname{sim}(s_i^+,{\boldsymbol x}_{k'})}{\tau} + \log\Bigg(\sum_{k=0}^{m} \exp\Big(\frac{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)}{\tau}\Big)\Bigg),
$$
where $\operatorname{sim}(s_i^+,{\boldsymbol x}) = \frac{s_i^+\cdot {\boldsymbol x}}{\|s_i^+\|\|{\boldsymbol x}\|}$ is the cosine similarity. The ${\boldsymbol x}_{k'}$ is a positive sample of the $s_i^+$.
Based on the Cauchy–Schwarz inequality and the Lipschitz property, for each $k$ there exists a constant $C>0$ such that:
$$
|\operatorname{sim}(s_i^+,{\boldsymbol x}_k) - \operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_k)| \le C \| s_i^+ - \mathfrak{s}_i^+ \|_2, \quad \forall i,k.
$$
The difference in NCE losses between $s_i^+$ and $\mathfrak{s}_i^+$ can be bounded as:
$$
\begin{aligned}
\big|\mathcal{L}_{\mathrm{NCE}}(s_i^+) - \mathcal{L}_{\mathrm{NCE}}(\mathfrak{s}_i^+)\big|
&\le \frac{|\operatorname{sim}(s_i^+,{\boldsymbol x}_0) - \operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_0)|}{\tau} \\
&+ \Bigg| \log \sum_{k=0}^m e^{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)/\tau} - \log \sum_{k=0}^m e^{\operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_k)/\tau} \Bigg|.
\end{aligned}
$$
Using the Lipschitz property of the $\log$-sum-exp function, the second term on the right-hand side can be further bounded by:
$$
\begin{aligned}
\Bigg| \log \sum_{k=0}^m e^{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)/\tau} - \log \sum_{k=0}^m e^{\operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_k)/\tau} \Bigg|
&\le \frac{\sum_{k=0}^m \big| e^{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)/\tau} - e^{\operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_k)/\tau} \big|}{\min(S_{s_i^+}, S_{\mathfrak{s}_i^+})}.
\end{aligned}
$$
where
$$
S_{s_i^+} = \sum_{k=0}^m e^{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)/\tau}, \quad S_{\mathfrak{s}_i^+} = \sum_{k=0}^m e^{\operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_k)/\tau}.
$$
Applying the cosine bound and Lagrange mean value theorem to the exponential terms yields:
$$
\big| e^{\operatorname{sim}(s_i^+,{\boldsymbol x}_k)/\tau} - e^{\operatorname{sim}(\mathfrak{s}_i^+,{\boldsymbol x}_k)/\tau} \big| \le \frac{C}{\tau} e^{1/\tau} \| s_i^+ - \mathfrak{s}_i^+ \|_2.
$$
Combining the previous inequalities, we obtain the Lipschitz bound for the $i$th anchor:
$$
\big|\mathcal{L}_{\mathrm{NCE}}(s_i^+) - \mathcal{L}_{\mathrm{NCE}}(\mathfrak{s}_i^+)\big| \le K \| s_i^+ - \mathfrak{s}_i^+ \|_2,
$$
where
$$
K = \frac{C}{\tau} \left( 1 + e^{2/\tau} \right).
$$
Finally, aggregating over all anchors $i=1,\dots,N$, we have:
$$
\big\| \mathcal{L}_{\mathrm{NCE}_s} - \mathcal{L}_{\mathrm{NCE}_\mathfrak{s}} \big\|_2 \le K \big\| s^+ - \mathbf{\mathfrak{s}}^+ \big\|_2,
$$
where $s^+ = [s_1^+,\dots,s_N^+]$ and $\mathbf{\mathfrak{s}}^+ = [\mathfrak{s}_1^+,\dots,\mathfrak{s}_N^+]$. $\blacksquare$
[^1]: Reference for representation learning bounds.
