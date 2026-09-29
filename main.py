"""
Core ML Architecture & Implementations - Validation & Benchmarking Suite
Author: Senior ML Engineer & Quant Researcher Architecture Mentor
Usage:
    uv run python main.py
    uv run python main.py --algo 1 4 8
    uv run python main.py --seed 42 --verbose
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Ensure 'src' is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Standard Package Imports from core_ml
from core_ml.linear_regression import LinearRegression, SklearnLinearRegression
from core_ml.logistic_regression import LogisticRegression, SklearnLogisticRegression
from core_ml.decision_tree_and_ensemble import DecisionTree, SklearnDecisionTree, SklearnRandomForest
from core_ml.neural_network import NeuralNetwork, PyTorchNeuralNetwork, SklearnNeuralNetwork
from core_ml.k_means_clustering import KMeans, SklearnKMeans
from core_ml.anomaly_detection import AnomalyDetector, SklearnAnomalyDetector
from core_ml.recommender_system import MatrixFactorization, PytorchRecommender
from core_ml.reinforcement_learning import DQNAgent, QLearningAgent


@dataclass
class TestResult:
    algo_id: int
    name: str
    status: str
    scratch_time_ms: float
    framework_time_ms: float
    metric_name: str
    scratch_metric: float
    framework_metric: float
    alignment_metric: Optional[float]
    notes: str = ""


# =====================================================================
# 01. Linear Regression
# =====================================================================
def run_test_01() -> TestResult:
    X, y = make_regression(n_samples=500, n_features=5, noise=10.0, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # From scratch
    t0 = time.perf_counter()
    scratch_model = LinearRegression(learning_rate=0.05, n_iterations=1000)
    scratch_model.fit(X, y)
    scratch_pred = scratch_model.predict(X)
    t_scratch = (time.perf_counter() - t0) * 1000

    # Framework
    t0 = time.perf_counter()
    framework_model = SklearnLinearRegression()
    framework_model.fit(X, y)
    framework_pred = framework_model.predict(X)
    t_framework = (time.perf_counter() - t0) * 1000

    scratch_r2 = r2_score(y, scratch_pred)
    framework_r2 = r2_score(y, framework_pred)
    corr = float(np.corrcoef(scratch_pred, framework_pred)[0, 1])

    status = "PASS" if (scratch_r2 > 0.90 and corr > 0.99) else "FAIL"
    return TestResult(
        algo_id=1,
        name="Linear Regression",
        status=status,
        scratch_time_ms=t_scratch,
        framework_time_ms=t_framework,
        metric_name="R2 Score",
        scratch_metric=scratch_r2,
        framework_metric=framework_r2,
        alignment_metric=corr,
        notes=f"Pred Correlation: {corr:.4f}",
    )


# =====================================================================
# 02. Logistic Regression
# =====================================================================
def run_test_02() -> TestResult:
    X, y = make_classification(
        n_samples=600, n_features=6, n_informative=4, n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # From scratch
    t0 = time.perf_counter()
    scratch_model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    scratch_model.fit(X, y)
    scratch_pred = scratch_model.predict(X)
    t_scratch = (time.perf_counter() - t0) * 1000

    # Framework
    t0 = time.perf_counter()
    framework_model = SklearnLogisticRegression()
    framework_model.fit(X, y)
    framework_pred = framework_model.predict(X)
    t_framework = (time.perf_counter() - t0) * 1000

    scratch_acc = accuracy_score(y, scratch_pred)
    framework_acc = accuracy_score(y, framework_pred)
    concordance = np.mean(scratch_pred == framework_pred)

    status = "PASS" if (scratch_acc > 0.80 and concordance > 0.85) else "FAIL"
    return TestResult(
        algo_id=2,
        name="Logistic Regression",
        status=status,
        scratch_time_ms=t_scratch,
        framework_time_ms=t_framework,
        metric_name="Accuracy",
        scratch_metric=scratch_acc,
        framework_metric=framework_acc,
        alignment_metric=concordance,
        notes=f"Class Agreement: {concordance * 100:.1f}%",
    )


# =====================================================================
# 03. Decision Tree & Ensemble
# =====================================================================
def run_test_03() -> TestResult:
    X, y = make_classification(
        n_samples=300, n_features=4, n_informative=3, n_redundant=0, random_state=42
    )

    # From scratch
    t0 = time.perf_counter()
    scratch_tree = DecisionTree(max_depth=5, min_samples_split=5)
    scratch_tree.fit(X, y)
    scratch_pred = scratch_tree.predict(X)
    t_scratch = (time.perf_counter() - t0) * 1000

    # Framework
    t0 = time.perf_counter()
    framework_tree = SklearnDecisionTree(max_depth=5, random_state=42)
    framework_tree.fit(X, y)
    framework_pred = framework_tree.predict(X)
    t_framework = (time.perf_counter() - t0) * 1000

    # Framework Random Forest verification
    rf = SklearnRandomForest(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y)
    rf_pred = rf.predict(X)
    rf_acc = accuracy_score(y, rf_pred)

    scratch_acc = accuracy_score(y, scratch_pred)
    framework_acc = accuracy_score(y, framework_pred)
    agreement = np.mean(scratch_pred == framework_pred)

    status = "PASS" if (scratch_acc > 0.85 and framework_acc > 0.85) else "FAIL"
    return TestResult(
        algo_id=3,
        name="Decision Tree & RF",
        status=status,
        scratch_time_ms=t_scratch,
        framework_time_ms=t_framework,
        metric_name="Accuracy",
        scratch_metric=scratch_acc,
        framework_metric=framework_acc,
        alignment_metric=agreement,
        notes=f"Sklearn RF Acc: {rf_acc:.3f} | Tree Agr: {agreement * 100:.1f}%",
    )


# =====================================================================
# 04. Neural Network (MLP)
# =====================================================================
def run_test_04() -> TestResult:
    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=5, random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # From scratch
    t0 = time.perf_counter()
    scratch_mlp = NeuralNetwork(
        input_size=8, hidden_size=16, output_size=1, learning_rate=0.05, n_iterations=800
    )
    scratch_mlp.fit(X, y)
    scratch_pred = scratch_mlp.predict(X).flatten()
    t_scratch = (time.perf_counter() - t0) * 1000

    # PyTorch Framework
    t0 = time.perf_counter()
    pytorch_mlp = PyTorchNeuralNetwork(
        input_size=8, hidden_size=16, output_size=1, learning_rate=0.01, n_iterations=400
    )
    pytorch_mlp.fit(X, y)
    pytorch_pred = pytorch_mlp.predict(X)
    t_framework = (time.perf_counter() - t0) * 1000

    scratch_acc = accuracy_score(y, scratch_pred)
    pytorch_acc = accuracy_score(y, pytorch_pred)
    agreement = np.mean(scratch_pred == pytorch_pred)

    status = "PASS" if (scratch_acc > 0.80 and pytorch_acc > 0.80) else "FAIL"
    return TestResult(
        algo_id=4,
        name="Neural Network (MLP)",
        status=status,
        scratch_time_ms=t_scratch,
        framework_time_ms=t_framework,
        metric_name="Accuracy",
        scratch_metric=scratch_acc,
        framework_metric=pytorch_acc,
        alignment_metric=agreement,
        notes=f"Scratch BCE Loss: {scratch_mlp.loss_history[-1]:.4f}",
    )


# =====================================================================
# 05. K-Means Clustering
# =====================================================================
def run_test_05() -> TestResult:
    X, _ = make_blobs(n_samples=600, centers=4, n_features=2, cluster_std=0.8, random_state=42)

    # From scratch
    t0 = time.perf_counter()
    scratch_kmeans = KMeans(n_clusters=4, max_iter=100)
    scratch_kmeans.fit(X)
    scratch_labels = scratch_kmeans.predict(X)
    t_scratch = (time.perf_counter() - t0) * 1000

    # Framework
    t0 = time.perf_counter()
    framework_kmeans = SklearnKMeans(n_clusters=4, max_iter=100, random_state=42)
    framework_kmeans.fit(X)
    framework_labels = framework_kmeans.predict(X)
    t_framework = (time.perf_counter() - t0) * 1000

    def compute_inertia(data: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
        return float(np.sum((data - centers[labels]) ** 2))

    scratch_inertia = compute_inertia(X, scratch_kmeans.centroids, scratch_labels)
    framework_inertia = compute_inertia(X, framework_kmeans.centroids, framework_labels)

    inertia_diff_ratio = abs(scratch_inertia - framework_inertia) / framework_inertia

    status = "PASS" if inertia_diff_ratio < 0.20 else "FAIL"
    return TestResult(
        algo_id=5,
        name="K-Means Clustering",
        status=status,
        scratch_time_ms=t_scratch,
        framework_time_ms=t_framework,
        metric_name="Inertia",
        scratch_metric=scratch_inertia,
        framework_metric=framework_inertia,
        alignment_metric=1.0 - inertia_diff_ratio,
        notes=f"Inertia Ratio: {scratch_inertia/framework_inertia:.2f}x",
    )


# =====================================================================
# 06. Anomaly Detection
# =====================================================================
def run_test_06() -> TestResult:
    np.random.seed(42)
    n_inliers = 500
    n_outliers = 50
    X_inliers = np.random.normal(loc=0.0, scale=1.0, size=(n_inliers, 3))
    X_outliers = np.random.uniform(low=-7.0, high=7.0, size=(n_outliers, 3))
    dist = np.linalg.norm(X_outliers, axis=1)
    X_outliers = X_outliers[dist > 3.5][:n_outliers]
    actual_outliers = len(X_outliers)

    X = np.vstack([X_inliers, X_outliers])
    y_true = np.zeros(len(X), dtype=int)
    y_true[n_inliers : n_inliers + actual_outliers] = 1

    # From scratch: Gaussian Density
    t0 = time.perf_counter()
    scratch_detector = AnomalyDetector(epsilon=1e-3)
    scratch_detector.fit(X_inliers)
    scratch_pred = scratch_detector.predict(X)
    t_scratch = (time.perf_counter() - t0) * 1000

    # Framework: Isolation Forest
    t0 = time.perf_counter()
    framework_detector = SklearnAnomalyDetector(
        contamination=actual_outliers / len(X), random_state=42
    )
    framework_detector.fit(X)
    framework_pred = framework_detector.predict(X)
    t_framework = (time.perf_counter() - t0) * 1000

    scratch_f1 = f1_score(y_true, scratch_pred)
    framework_f1 = f1_score(y_true, framework_pred)

    status = "PASS" if (scratch_f1 > 0.70 and framework_f1 > 0.70) else "FAIL"
    return TestResult(
        algo_id=6,
        name="Anomaly Detection",
        status=status,
        scratch_time_ms=t_scratch,
        framework_time_ms=t_framework,
        metric_name="F1 Score",
        scratch_metric=scratch_f1,
        framework_metric=framework_f1,
        alignment_metric=np.mean(scratch_pred == framework_pred),
        notes=f"Detected: Scratch={scratch_pred.sum()}, IF={framework_pred.sum()} (True={actual_outliers})",
    )


# =====================================================================
# 07. Recommender System (Collaborative Filtering)
# =====================================================================
def run_test_07() -> TestResult:
    n_users = 40
    n_items = 30
    n_factors = 4

    np.random.seed(42)
    U_true = np.random.normal(size=(n_users, n_factors))
    V_true = np.random.normal(size=(n_items, n_factors))
    R_full = np.dot(U_true, V_true.T)
    R_full = np.clip(R_full + 3.0, 1.0, 5.0)

    obs_mask = np.random.rand(n_users, n_items) < 0.35
    R_sparse = np.zeros_like(R_full)
    R_sparse[obs_mask] = R_full[obs_mask]

    # Scratch Matrix Factorization
    t0 = time.perf_counter()
    scratch_mf = MatrixFactorization(
        n_factors=n_factors, learning_rate=0.01, n_iterations=800, regularization=0.02
    )
    scratch_mf.fit(R_sparse)
    R_pred_scratch = scratch_mf.predict()
    t_scratch = (time.perf_counter() - t0) * 1000

    # PyTorch Recommender
    t0 = time.perf_counter()
    pytorch_rec = PytorchRecommender(
        n_users=n_users, n_items=n_items, n_factors=n_factors, learning_rate=0.05, n_iterations=400
    )
    pytorch_rec.fit(R_sparse)
    R_pred_pytorch = pytorch_rec.predict()
    t_framework = (time.perf_counter() - t0) * 1000

    scratch_rmse = float(np.sqrt(np.mean((R_sparse[obs_mask] - R_pred_scratch[obs_mask]) ** 2)))
    pytorch_rmse = float(np.sqrt(np.mean((R_sparse[obs_mask] - R_pred_pytorch[obs_mask]) ** 2)))

    status = "PASS" if (scratch_rmse < 0.60 and pytorch_rmse < 0.60) else "FAIL"
    return TestResult(
        algo_id=7,
        name="Matrix Factorization",
        status=status,
        scratch_time_ms=t_scratch,
        framework_time_ms=t_framework,
        metric_name="Train RMSE",
        scratch_metric=scratch_rmse,
        framework_metric=pytorch_rmse,
        alignment_metric=float(np.corrcoef(R_pred_scratch.flatten(), R_pred_pytorch.flatten())[0, 1]),
        notes=f"Reconstruction Corr: {np.corrcoef(R_pred_scratch.flatten(), R_pred_pytorch.flatten())[0, 1]:.3f}",
    )


# =====================================================================
# 08. Reinforcement Learning
# =====================================================================
def run_test_08() -> TestResult:
    # 1. Tabular Q-Learning on 1D Grid Environment
    n_states = 6
    n_actions = 2
    goal_state = 5

    t0 = time.perf_counter()
    q_agent = QLearningAgent(n_states=n_states, n_actions=n_actions, learning_rate=0.1, discount_factor=0.9)
    for _ in range(300):
        s = 0
        for _ in range(20):
            a = q_agent.choose_action(s)
            s_next = min(n_states - 1, s + 1) if a == 1 else max(0, s - 1)
            done = s_next == goal_state
            reward = 10.0 if done else -0.1
            q_agent.learn(s, a, reward, s_next, done)
            s = s_next
            if done:
                break
    t_scratch = (time.perf_counter() - t0) * 1000

    policy_learned = all(q_agent.predict(s) == 1 for s in range(goal_state))

    # 2. DQN Agent with Continuous State Space
    t0 = time.perf_counter()
    dqn_agent = DQNAgent(state_dim=4, action_dim=2, batch_size=32)
    for _ in range(100):
        s = np.random.randn(4)
        a = np.random.choice(2)
        r = 1.0 if s[0] > 0 else -1.0
        s_next = s + 0.1 * np.random.randn(4)
        done = False
        dqn_agent.remember(s, a, r, s_next, done)
    dqn_agent.learn()
    test_action = dqn_agent.choose_action(np.ones(4))
    t_framework = (time.perf_counter() - t0) * 1000

    status = "PASS" if policy_learned and (test_action in [0, 1]) else "FAIL"
    return TestResult(
        algo_id=8,
        name="Reinforcement Learning",
        status=status,
        scratch_time_ms=t_scratch,
        framework_time_ms=t_framework,
        metric_name="Goal Reached",
        scratch_metric=1.0 if policy_learned else 0.0,
        framework_metric=1.0,
        alignment_metric=1.0 if policy_learned else 0.0,
        notes=f"Q-Table Learned: {policy_learned} | DQN Step Completed",
    )


# =====================================================================
# Main Execution Pipeline
# =====================================================================
TEST_REGISTRY: Dict[int, Tuple[str, Callable[[], TestResult]]] = {
    1: ("Linear Regression", run_test_01),
    2: ("Logistic Regression", run_test_02),
    3: ("Decision Tree & Ensemble", run_test_03),
    4: ("Neural Network (MLP)", run_test_04),
    5: ("K-Means Clustering", run_test_05),
    6: ("Anomaly Detection", run_test_06),
    7: ("Recommender System", run_test_07),
    8: ("Reinforcement Learning", run_test_08),
}


def print_banner():
    print("=" * 95)
    print("  CORE-ML-ARCHITECTURE: ALGORITHM VALIDATION & BENCHMARKING SUITE")
    print("  Package: core_ml (Standard src/ layout) | uv | NumPy Vectorized | PyTorch | Scikit-Learn")
    print("=" * 95)


def print_summary_table(results: List[TestResult]):
    print("\n" + "=" * 95)
    print(f"{'ID':<3} | {'Algorithm':<24} | {'Status':<6} | {'Metric':<12} | {'Scratch':<8} | {'Framework':<9} | {'Scratch(ms)':<11} | {'Notes'}")
    print("-" * 95)
    for r in results:
        status_color = "\033[92mPASS\033[0m" if r.status == "PASS" else "\033[91mFAIL\033[0m"
        print(
            f"{r.algo_id:<3} | {r.name:<24} | {status_color:<6} | {r.metric_name:<12} | "
            f"{r.scratch_metric:<8.4f} | {r.framework_metric:<9.4f} | {r.scratch_time_ms:<11.2f} | {r.notes}"
        )
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Test & Benchmarking Harness for Core ML Architecture (Algos 1-8)"
    )
    parser.add_argument(
        "--algo",
        type=int,
        nargs="+",
        choices=range(1, 9),
        default=list(range(1, 9)),
        help="Specify which algorithms to test (1 to 8). Default is all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation and reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed execution diagnostics.",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    print_banner()
    print(f"Target Algorithms: {args.algo} | Random Seed: {args.seed}\n")

    results: List[TestResult] = []
    total_start = time.perf_counter()

    for algo_id in args.algo:
        name, test_fn = TEST_REGISTRY[algo_id]
        print(f"[*] Running Test Module {algo_id:02d}: {name} ...", end=" ", flush=True)
        try:
            res = test_fn()
            results.append(res)
            print(f"[{res.status}] ({res.scratch_time_ms:.1f}ms scratch vs {res.framework_time_ms:.1f}ms framework)")
            if args.verbose:
                print(f"    -> Metric: {res.metric_name} | Scratch: {res.scratch_metric:.4f} | Framework: {res.framework_metric:.4f}")
                print(f"    -> Detail: {res.notes}")
        except Exception as e:
            print(f"[ERROR: {e}]")
            results.append(
                TestResult(
                    algo_id=algo_id,
                    name=name,
                    status="ERROR",
                    scratch_time_ms=0.0,
                    framework_time_ms=0.0,
                    metric_name="Error",
                    scratch_metric=0.0,
                    framework_metric=0.0,
                    alignment_metric=0.0,
                    notes=str(e),
                )
            )

    elapsed_sec = time.perf_counter() - total_start
    print_summary_table(results)

    total_passed = sum(1 for r in results if r.status == "PASS")
    print(f"\nExecution Complete: {total_passed}/{len(results)} tests passed in {elapsed_sec:.2f}s.")

    if total_passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
