# Theoretic Motivation

Let the input $\boldsymbol{X}$ contain two underlying generative factors:

* a **semantic factor** reflecting the turbine operation, $\mathfrak{s}\in\mathcal{S}$ (e.g., operating mode);
* a **detail factor** $\mathfrak{g}\in\mathcal{G}$ (e.g., sensor-level variations or noise).

Let the encoder produce a normalized latent representation $\boldsymbol{z}$. The semantic mutual information is defined as


I(\boldsymbol{z}; \mathfrak{s})
===============================
$$
\mathbb{E}_{p(\boldsymbol{z},\mathfrak{s})}
\left[
\log
\frac{p(\boldsymbol{z},\mathfrak{s})}
{p(\boldsymbol{z})p(\mathfrak{s})}
\right],
$$

which measures how much semantic content $\boldsymbol{z}$ retains.

Similarly, the conditional mutual information

$$
I(\boldsymbol{z}; \mathfrak{g}\mid \mathfrak{s})
$$

measures the amount of detail information contained in $\boldsymbol{z}$ after accounting for the semantic factor $\mathfrak{s}$.

Therefore, a desirable encoder should:

* maximize $I(\boldsymbol{z};\mathfrak{s})$;
* minimize $I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s})$;

thereby alleviating the **semantic leakage (SL)** problem.

---

## Theorem 1: Semantic Enhancement and Detail Suppression

**Theorem.**
Let $\boldsymbol{z}\in\mathbb{R}^d$ denote the normalized representation produced by the encoder, i.e.,

$$
|\boldsymbol{z}|_2 = 1,
$$

and consider a contrastive learning framework with $N$ anchor samples.

Then, the conditional mutual information between $\boldsymbol{z}$ and the detail factor $\mathfrak{g}$ satisfies

$$
I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s})
\le
C-\log N+\mathcal{L}*{\mathrm{NCE}*{\mathfrak{s}}},
$$

and

$$
I(\boldsymbol{z};\mathfrak{s})
\ge
\log N-\mathcal{L}*{\mathrm{NCE}*{\mathfrak{s}}},
$$

where

$$
\mathcal{L}*{\mathrm{NCE}*{\mathfrak{s}}}
=========================================

-\mathbb{E}*i
\left[
\log
\frac{
\exp\left(
\operatorname{sim}(\mathfrak{s}*{i'},\boldsymbol{z}*i)
\right)
}{
\sum*{j=1}^{N}
\exp\left(
\operatorname{sim}(\mathfrak{s}_j,\boldsymbol{z}_i)
\right)
}
\right],
$$

with $i'=1,\dots,N$, is an InfoNCE-type contrastive loss induced by a mode-matching mechanism under ideal wind-turbine semantics.

The constant $C$ is given by

$$
C
=

\log
\frac{2\pi^{d/2}}
{\Gamma(d/2)},
$$

where $\Gamma(\cdot)$ denotes the Gamma function.

---

## Approximate Semantic Representation

Furthermore, suppose that the learned semantic representation $s$ satisfies

$$
|s-\mathfrak{s}|_2\le\varepsilon,
$$

for a sufficiently small $\varepsilon>0$.

Then, there exists a constant $K>0$ such that

$$
\left|
\mathcal{L}_{\mathrm{NCE}_s}
----------------------------

\mathcal{L}*{\mathrm{NCE}*{\mathfrak{s}}}
\right|_2
\le
K|s-\mathfrak{s}|_2.
$$

Consequently,

$$
I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s})
\le
C-\log N
+
\mathcal{L}_{\mathrm{NCE}_s}
\pm
K|s-\mathfrak{s}|_2.
$$

Thus, minimizing the contrastive loss $\mathcal{L}_{\mathrm{NCE}_s}$ associated with the learned semantic representation can increase the mutual information

$$
I(\boldsymbol{z};\mathfrak{s})
$$

while tightening the upper bound of

$$
I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s}).
$$

This provides a theoretical justification for learning representations that increasingly preserve semantic structure while suppressing irrelevant detail information.

---

# Proofs

## Appendix I: Proof of Theorem 1

According to the definition of $\mathcal{L}*{\mathrm{NCE}*{\mathfrak{s}_i}}$ and the InfoNCE lower bound [1], the mutual information between $\boldsymbol{z}$ and the semantic factor $\mathfrak{s}_i$ satisfies

$$
I(\boldsymbol{z};\mathfrak{s}_i)
\ge
\log N
------

\mathcal{L}*{\mathrm{NCE}*{\mathfrak{s}_i}}.
$$

By the chain rule of mutual information,

$$
I(\boldsymbol{z};(\mathfrak{s}_i,\mathfrak{g}))
===============================================

# I(\boldsymbol{z};\boldsymbol{X})

I(\boldsymbol{z};\mathfrak{s}_i)
+
I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s}_i).
$$

Since

$$
I(\boldsymbol{z};\boldsymbol{X})
\le
H(\boldsymbol{z}),
$$

we obtain

$$
I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s}_i)
\le
H(\boldsymbol{z})
-----------------

I(\boldsymbol{z};\mathfrak{s}_i).
$$

Substituting the InfoNCE lower bound gives

$$
I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s}_i)
\le
H(\boldsymbol{z})
-----------------

\log N
+
\mathcal{L}*{\mathrm{NCE}*{\mathfrak{s}_i}}.
$$

Since $|\boldsymbol{z}|_2=1$, $\boldsymbol{z}$ lies on the unit hypersphere in $\mathbb{R}^d$. Its entropy is therefore bounded by the logarithm of the surface area of the unit hypersphere:

$$
H(\boldsymbol{z})
\le
\log
\frac{2\pi^{d/2}}
{\Gamma(d/2)}
=============

C.
$$

Hence,

$$
I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s}*i)
\le
C-\log N
+
\mathcal{L}*{\mathrm{NCE}_{\mathfrak{s}_i}}.
$$

Since the inequality holds for all $i=1,\dots,N$, aggregating over the semantic samples yields

$$
I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s})
\le
C-\log N
+
\mathcal{L}*{\mathrm{NCE}*{\mathfrak{s}}}.
$$

The lower bound

$$
I(\boldsymbol{z};\mathfrak{s})
\ge
\log N-\mathcal{L}*{\mathrm{NCE}*{\mathfrak{s}}}
$$

follows directly from the InfoNCE inequality.

Therefore, minimizing $\mathcal{L}*{\mathrm{NCE}*{\mathfrak{s}}}$ simultaneously encourages semantic information preservation and suppresses detail information in the learned representation.

$\square$

---

## Appendix II: Proof of the Lipschitz Inequality

Let

$$
{s_i^+}_{i=1}^{N}
\quad\text{and}\quad
{\mathfrak{s}*i^+}*{i=1}^{N}
$$

be two sets of anchor vectors from different semantic Hilbert spaces, with corresponding positive and negative samples

$$
{\boldsymbol{x}*k}*{k=0}^{m}.
$$

Assume that all vectors have norms bounded away from zero:

$$
|s_i^+|,
|\mathfrak{s}_i^+|,
|\boldsymbol{x}_k|
\ge
c>0,
\qquad
\forall i,k.
$$

Let $\tau>0$ denote the temperature parameter.

The NCE loss for anchor $s_i^+$ is defined as

$$
\mathcal{L}_{\mathrm{NCE}}(s_i^+)
=================================

-\frac{
\operatorname{sim}(s_i^+,\boldsymbol{x}*{k'})
}{\tau}
+
\log
\left(
\sum*{k=0}^{m}
\exp
\left(
\frac{
\operatorname{sim}(s_i^+,\boldsymbol{x}_k)
}{\tau}
\right)
\right),
$$

where

$$
\operatorname{sim}(s_i^+,\boldsymbol{x})
========================================

\frac{
s_i^+\cdot\boldsymbol{x}
}{
|s_i^+|,|\boldsymbol{x}|
}
$$

is the cosine similarity, and $\boldsymbol{x}_{k'}$ is a positive sample associated with $s_i^+$.

### Step 1: Lipschitz Continuity of Cosine Similarity

Based on the Cauchy--Schwarz inequality and the Lipschitz continuity of normalized inner products, there exists a constant $C>0$ such that

$$
\left|
\operatorname{sim}(s_i^+,\boldsymbol{x}_k)
------------------------------------------

\operatorname{sim}(\mathfrak{s}_i^+,\boldsymbol{x}_k)
\right|
\le
C
|s_i^+-\mathfrak{s}_i^+|_2,
\qquad
\forall i,k.
$$

### Step 2: Difference Between the Two NCE Losses

The difference between the NCE losses can be bounded as

$$
\begin{aligned}
&
\left|
\mathcal{L}_{\mathrm{NCE}}(s_i^+)
---------------------------------

\mathcal{L}_{\mathrm{NCE}}(\mathfrak{s}_i^+)
\right|
\
&\le
\frac{
\left|
\operatorname{sim}(s_i^+,\boldsymbol{x}_0)
------------------------------------------

\operatorname{sim}(\mathfrak{s}_i^+,\boldsymbol{x}*0)
\right|
}{\tau}
\
&\quad+
\left|
\log
\sum*{k=0}^{m}
e^{\operatorname{sim}(s_i^+,\boldsymbol{x}_k)/\tau}
---------------------------------------------------

\log
\sum_{k=0}^{m}
e^{\operatorname{sim}(\mathfrak{s}_i^+,\boldsymbol{x}_k)/\tau}
\right|.
\end{aligned}
$$

Define

$$
S_{s_i^+}
=========

\sum_{k=0}^{m}
e^{\operatorname{sim}(s_i^+,\boldsymbol{x}_k)/\tau},
$$

and

$$
S_{\mathfrak{s}_i^+}
====================

\sum_{k=0}^{m}
e^{\operatorname{sim}(\mathfrak{s}_i^+,\boldsymbol{x}_k)/\tau}.
$$

Using the Lipschitz property of the logarithm and the positivity of the log-sum-exp arguments,

$$
\begin{aligned}
&
\left|
\log S_{s_i^+}
--------------

\log S_{\mathfrak{s}*i^+}
\right|
\
&\le
\frac{
\displaystyle
\sum*{k=0}^{m}
\left|
e^{\operatorname{sim}(s_i^+,\boldsymbol{x}_k)/\tau}
---------------------------------------------------

e^{\operatorname{sim}(\mathfrak{s}*i^+,\boldsymbol{x}*k)/\tau}
\right|
}{
\min(S*{s_i^+},S*{\mathfrak{s}_i^+})
}.
\end{aligned}
$$

### Step 3: Bounding the Exponential Terms

Since cosine similarity satisfies

$$
-1
\le
\operatorname{sim}(\cdot,\cdot)
\le
1,
$$

the mean value theorem for the exponential function gives

$$
\begin{aligned}
&
\left|
e^{\operatorname{sim}(s_i^+,\boldsymbol{x}_k)/\tau}
---------------------------------------------------

e^{\operatorname{sim}(\mathfrak{s}_i^+,\boldsymbol{x}_k)/\tau}
\right|
\
&\le
\frac{C}{\tau}
e^{1/\tau}
|s_i^+-\mathfrak{s}_i^+|_2.
\end{aligned}
$$

Combining the above inequalities yields

$$
\left|
\mathcal{L}_{\mathrm{NCE}}(s_i^+)
---------------------------------

\mathcal{L}_{\mathrm{NCE}}(\mathfrak{s}_i^+)
\right|
\le
K
|s_i^+-\mathfrak{s}_i^+|_2,
$$

where

$$
K
=

\frac{C}{\tau}
\left(
1+e^{2/\tau}
\right).
$$

Finally, aggregating over all anchors $i=1,\dots,N$, we obtain

$$
\left|
\mathcal{L}_{\mathrm{NCE}_s}
----------------------------

\mathcal{L}*{\mathrm{NCE}*{\mathfrak{s}}}
\right|_2
\le
K
\left|
s^+
---

\boldsymbol{\mathfrak{s}}^+
\right|_2,
$$

where

$$
s^+
===

[s_1^+,\dots,s_N^+],
$$

and

$$
\boldsymbol{\mathfrak{s}}^+
===========================

[\mathfrak{s}_1^+,\dots,\mathfrak{s}_N^+].
$$

Therefore, the NCE objective is Lipschitz continuous with respect to the semantic representation, completing the proof.

$\square$

---

## Summary

The theoretical results establish the following relationship:

$$
\boxed{
\mathcal{L}_{\mathrm{NCE}_s}\downarrow
\quad\Longrightarrow\quad
I(\boldsymbol{z};\mathfrak{s})\uparrow
}
$$

and

$$
\boxed{
\mathcal{L}_{\mathrm{NCE}_s}\downarrow
\quad\Longrightarrow\quad
I(\boldsymbol{z};\mathfrak{g}\mid\mathfrak{s})\downarrow
}
$$

up to the approximation error

$$
K|s-\mathfrak{s}|_2.
$$

Hence, semantic-aware contrastive learning provides a principled mechanism for **semantic enhancement and detail suppression**, which is the theoretical motivation for mitigating semantic leakage in the learned representation.

## Reference

[1]contrastive predictive coding."arXiv, 2018 Representation Jearning withk
