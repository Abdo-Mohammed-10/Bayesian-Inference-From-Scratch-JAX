# Bayesian CNN with JAX & Flax: Uncertainty Estimation from Scratch 🧠📊
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15BGjg5EYJRYH8drX4jkd38fzSu-lahLx?usp=sharing)
[![JAX](https://img.shields.io/badge/JAX-Accelerated%20Linear%20Algebra-blue)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-Neural%20Network%20Library-green)](https://github.com/google/flax)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


> **A probabilistic deep learning implementation building Bayesian Neural Networks (BNNs) from scratch using JAX and Flax. This project demonstrates uncertainty estimation and Out-of-Distribution (OOD) detection.**

---

## 🚀 Overview

Standard Deep Neural Networks (DNNs) are often "overconfident" even when they are wrong. In safety-critical applications (e.g., medical imaging, autonomous driving), knowing *what the model doesn't know* is as important as accuracy.

This project implements a **Variational Inference (VI)** approach to Bayesian Neural Networks. Unlike standard libraries, the Bayesian Dense layers are implemented **from scratch** to demonstrate a deep understanding of the underlying mathematics (ELBO, KL Divergence, and the Reparameterization Trick).

### Key Features
* **Custom Bayesian Layers:** Implemented `VariationalDense` layer manually inheriting from `flax.linen.Module`.
* **Variational Inference:** Optimization using the **Evidence Lower Bound (ELBO)** loss function.
* **JAX Power:** Utilizing `jax.vmap` for efficient Monte Carlo sampling during inference.
* **Uncertainty Quantification:** Measuring **Predictive Entropy** to distinguish between confident and uncertain predictions.
* **Reliability Analysis:** Visualizing Expected Calibration Error (ECE) and Reliability Diagrams.
* **OOD Detection:** Evaluation on Out-of-Distribution data (Noise/Contrast) vs. In-Distribution (FashionMNIST).

---

## 🛠️ Tech Stack & Methodology

* **Frameworks:** `JAX`, `Flax`, `Optax`
* **Data Pipeline:** `TensorFlow Datasets (TFDS)`
* **Visualization:** `Matplotlib`, `Seaborn`

### Mathematical Foundation
The model approximates the posterior distribution $p(w|D)$ using a variational distribution $q_\theta(w)$. The training minimizes the **Negative ELBO**:

$$\mathcal{L}(\theta) = \underbrace{D_{KL}(q_\theta(w) \parallel p(w))}_{\text{Complexity Cost}} - \underbrace{\mathbb{E}_{q_\theta}[\log p(y|x, w)]}_{\text{Likelihood Cost}}$$

I implemented the **Reparameterization Trick** for the weights:
$$w = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

---

## 📊 Results & Visualization

*(You can insert your generated plots here)*

### 1. Uncertainty Estimation (OOD Detection)
The model successfully assigns higher **Predictive Entropy** to Out-of-Distribution samples compared to known data.

| Data Source | Avg. Entropy (Uncertainty) | Interpretation |
| :--- | :--- | :--- |
| **In-Distribution** | **Low** | Model is confident. |
| **Out-of-Distribution** | **High** | Model correctly flags ignorance. |

### 2. Reliability Diagram
The **Expected Calibration Error (ECE)** analysis shows how closely the predicted probabilities align with actual accuracy.

---

## 💻 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/JAX-Bayesian-From-Scratch.git](https://github.com/YourUsername/JAX-Bayesian-From-Scratch.git)
   cd JAX-Bayesian-From-Scratch
