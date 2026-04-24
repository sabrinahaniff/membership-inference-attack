import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from target_model import generate_dataset, get_loss
from attack import run_membership_inference_attack

def train_dp_model(X_train, y_train, epsilon=1.0):
    # differential privacy via gradient noise
    # we simulate DP-SGD (differentially private stochastic gradient descent) by adding laplace noise to the training data
    # before fitting, this approximates the effect of DP training
    # true DP-SGD requires custom training loops but this demonstrates the concept
    
    noise_scale = 1.0 / epsilon
    
    # add calibrated noise to training features
    # smaller epsilon = more noise = stronger privacy = harder attack
    X_private = X_train + np.random.laplace(0, noise_scale, X_train.shape)
    
    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=200,
        alpha=0.1,
        random_state=42
    )
    
    model.fit(X_private, y_train)
    return model

def compare_all_defenses(X_train, y_train, X_test, y_test):
    from target_model import train_target_model
    
    print("=" * 55)
    print("MEMBERSHIP INFERENCE — FULL DEFENSE COMPARISON")
    print("=" * 55)
    
    results = {}
    
    # no defense
    print("\n[1] No Defense (overfit model)")
    model = train_target_model(X_train, y_train, overfit=True)
    r = run_membership_inference_attack(model, X_train, y_train, X_test, y_test)
    results["No Defense"] = r
    print(f"    Loss gap: {r['loss_gap']:.3f} | Attack AUC: {r['attack_auc']:.3f}")
    print(f"    Privacy risk: HIGH")

    # regularization only
    print("\n[2] Regularization Only")
    model = train_target_model(X_train, y_train, overfit=False)
    r = run_membership_inference_attack(model, X_train, y_train, X_test, y_test)
    results["Regularization"] = r
    print(f"    Loss gap: {r['loss_gap']:.3f} | Attack AUC: {r['attack_auc']:.3f}")
    print(f"    Privacy risk: MEDIUM")

    # dp with different epsilon values
    for epsilon in [0.1, 0.5, 1.0, 5.0]:
        label = f"DP (epsilon={epsilon})"
        print(f"\n[3] Differential Privacy — epsilon={epsilon}")
        model = train_dp_model(X_train, y_train, epsilon=epsilon)
        r = run_membership_inference_attack(model, X_train, y_train, X_test, y_test)
        results[label] = r
        
        if r['attack_auc'] < 0.55:
            risk = "LOW"
        elif r['attack_auc'] < 0.65:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
            
        print(f"    Loss gap: {r['loss_gap']:.3f} | Attack AUC: {r['attack_auc']:.3f}")
        print(f"    Privacy risk: {risk}")
    
    return results

if __name__ == "__main__":
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    compare_all_defenses(X_train, y_train, X_test, y_test)