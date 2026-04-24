import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from target_model import generate_dataset, train_target_model
from attack import run_membership_inference_attack
from defense import train_dp_model

def run_all_experiments(X_train, y_train, X_test, y_test):
    results = {}

    # no defense: baseline to compare everything against
    model = train_target_model(X_train, y_train, overfit=True)
    results["No Defense"] = run_membership_inference_attack(
        model, X_train, y_train, X_test, y_test)

    # regularization only
    model = train_target_model(X_train, y_train, overfit=False)
    results["Regularization"] = run_membership_inference_attack(
        model, X_train, y_train, X_test, y_test)

    # dp at different epsilon values: want to see where the attack breaks down
    for epsilon in [0.1, 0.5, 1.0, 5.0]:
        model = train_dp_model(X_train, y_train, epsilon=epsilon)
        results[f"DP e={epsilon}"] = run_membership_inference_attack(
            model, X_train, y_train, X_test, y_test)

    return results

def plot_results(results):
    labels = list(results.keys())
    aucs = [results[k]["attack_auc"] for k in labels]
    gaps = [abs(results[k]["loss_gap"]) for k in labels]

    # color code by defense type
    colors = []
    for k in labels:
        if k == "No Defense":
            colors.append("#ef4444")      # red: dangerous!!! 
        elif k == "Regularization":
            colors.append("#f97316")      # orange: partial defense
        else:
            colors.append("#22c55e")      # green: dp works yay

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0e0e10")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#111116")
        ax.tick_params(colors="#94a3b8")
        ax.spines["bottom"].set_color("#1e1e24")
        ax.spines["top"].set_color("#1e1e24")
        ax.spines["left"].set_color("#1e1e24")
        ax.spines["right"].set_color("#1e1e24")

    # chart 1: attack AUC across all defenses
    bars = ax1.bar(labels, aucs, color=colors, alpha=0.85, width=0.6)
    ax1.axhline(y=0.5, color="#818cf8", linestyle="--",
                linewidth=1.5, label="random guessing (0.5)")
    ax1.set_ylabel("Attack AUC", color="#94a3b8")
    ax1.set_title("Attack AUC by Defense\nlower = better privacy",
                  color="#e2e8f0", pad=12)
    ax1.set_ylim(0, 1)
    ax1.legend(facecolor="#1e1e24", labelcolor="#94a3b8")
    ax1.set_xticklabels(labels, rotation=25, ha="right", color="#94a3b8")

    for bar, val in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                color="#e2e8f0", fontsize=9)

    # chart 2: loss gap across all defenses
    # this shows why the attack works or doesn't
    bars2 = ax2.bar(labels, gaps, color=colors, alpha=0.85, width=0.6)
    ax2.set_ylabel("Loss Gap", color="#94a3b8")
    ax2.set_title("Loss Gap by Defense\nsmaller gap = harder attack",
                  color="#e2e8f0", pad=12)
    ax2.set_xticklabels(labels, rotation=25, ha="right", color="#94a3b8")

    for bar, val in zip(bars2, gaps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                color="#e2e8f0", fontsize=9)

    plt.suptitle("Membership Inference Attack — Defense Comparison",
                color="#f1f5f9", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight",
                facecolor="#0e0e10")
    plt.show()
    print("saved to results.png")

if __name__ == "__main__":
    print("generating dataset...")
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("running experiments...")
    results = run_all_experiments(X_train, y_train, X_test, y_test)

    print("\n--- results ---")
    for name, r in results.items():
        print(f"{name:20s} | AUC: {r['attack_auc']:.3f} | gap: {r['loss_gap']:.3f}")

    print("\nplotting...")
    plot_results(results)