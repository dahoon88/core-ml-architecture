# Core ML Architecture & Implementations

## 📌 Overview
This repository provides production-grade, object-oriented implementations of core machine learning and deep learning algorithms designed for quantitative research and algorithmic trading engineering.

Each model is implemented in two parallel paradigms:
1. **From Scratch**: Pure NumPy matrix and vector operations without high-level ML libraries, emphasizing numerical stability, vectorization, and mathematical derivation.
2. **Framework Reference**: Industry-standard implementations utilizing **PyTorch** and **Scikit-Learn** for runtime benchmarking and precision validation.

---

## 📂 Project Architecture

The project adheres to the standard Python `src/` package layout:

```text
ml-dl/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated GitHub Actions test pipeline
├── src/
│   └── core_ml/
│       ├── __init__.py
│       ├── linear_regression/
│       │   ├── __init__.py
│       │   ├── from_scratch.py  # Gradient descent with vectorized MSE
│       │   └── framework.py     # Scikit-Learn wrapper
│       ├── logistic_regression/
│       │   ├── __init__.py
│       │   ├── from_scratch.py  # Binary cross-entropy & sigmoid
│       │   └── framework.py     # Scikit-Learn wrapper
│       ├── decision_tree_and_ensemble/
│       │   ├── __init__.py
│       │   ├── from_scratch.py  # Information gain & recursive partitioning
│       │   └── framework.py     # Sklearn DecisionTree & RandomForest
│       ├── neural_network/
│       │   ├── __init__.py
│       │   ├── from_scratch.py  # 2-layer MLP with backprop & ReLU/Sigmoid
│       │   └── framework.py     # PyTorch & Sklearn MLP implementations
│       ├── k_means_clustering/
│       │   ├── __init__.py
│       │   ├── from_scratch.py  # Vectorized distance broadcasting & centroid update
│       │   └── framework.py     # Scikit-Learn KMeans wrapper
│       ├── anomaly_detection/
│       │   ├── __init__.py
│       │   ├── from_scratch.py  # Multivariate Gaussian log-density estimation
│       │   └── framework.py     # Isolation Forest wrapper
│       ├── recommender_system/
│       │   ├── __init__.py
│       │   ├── from_scratch.py  # Vectorized matrix factorization (SGD)
│       │   └── framework.py     # PyTorch nn.Embedding latent factor model
│       └── reinforcement_learning/
│           ├── __init__.py
│           ├── from_scratch.py  # Tabular Bellman Q-Learning
│           └── framework.py     # Deep Q-Network (DQN) with PyTorch Experience Replay
├── main.py                      # Benchmarking & validation test harness
├── pyproject.toml               # uv / build configuration
├── uv.lock
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Management (`uv`)
Install dependencies via `uv`:
```bash
uv sync
```

### 2. Running the Test & Benchmarking Suite
Validate and benchmark all 8 algorithms:
```bash
uv run python main.py
```

Run specific algorithm modules (e.g., 1, 4, 8):
```bash
uv run python main.py --algo 1 4 8 --verbose
```

### 3. Using as an Imported Library
```python
from core_ml.linear_regression import LinearRegression
from core_ml.neural_network import PyTorchNeuralNetwork
from core_ml.reinforcement_learning import DQNAgent

# Initialize and fit
model = LinearRegression(learning_rate=0.01, n_iterations=500)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## ⚙️ CI/CD (Continuous Integration)

Automated testing is configured via **GitHub Actions** in `.github/workflows/ci.yml`. On every `push` and `pull_request` to `main`:
- Sets up Python 3.13 and caches dependencies using `astral-sh/setup-uv`.
- Installs the virtual environment using `uv sync`.
- Runs `uv run python main.py --verbose --seed 42` to verify zero regression across all 8 algorithms.
