import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from target_model import generate_dataset, get_loss
from attack import run_membership_inference_attack

def train_dp_model(X_train, y_train, epsilon=1.0):
    # simulating differential privacy by adding laplace noise to the training data
    # this isn't true DP-SGD (which requires per-sample gradient clipping)
    # same idea that noise prevents memorization
    
    # smaller epsilon = more noise = stronger privacy = harder attack
    # larger epsilon = less noise = weaker privacy = easier attack
    noise_scale = 1.0 / epsilon
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
    print("full defense comparison")
    print("=" * 55)
    print("(AUC of 0.5 = random guessing = attacker has no signal)")
    
    results = {}
    
    # baseline: no defense
    print("\n[1] no defense")
    model = train_target_model(X_train, y_train, overfit=True)
    r = run_membership_inference_attack(model, X_train, y_train, X_test, y_test)
    results["No Defense"] = r
    print(f"  loss gap: {r['loss_gap']:.3f}  |  AUC: {r['attack_auc']:.3f}  |  risk: HIGH")

    # regularization only
    print("\n[2] regularization only")
    model = train_target_model(X_train, y_train, overfit=False)
    r = run_membership_inference_attack(model, X_train, y_train, X_test, y_test)
    results["Regularization"] = r
    print(f"  loss gap: {r['loss_gap']:.3f}  |  AUC: {r['attack_auc']:.3f}  |  risk: MEDIUM")
    print("  helps but attacker still has some signal")

    # differential privacy at different epsilon values
    # curious to see where the attack actually breaks down
    for epsilon in [0.1, 0.5, 1.0, 5.0]:
        label = f"DP (epsilon={epsilon})"
        print(f"\n[3] differential privacy — epsilon={epsilon}")
        model = train_dp_model(X_train, y_train, epsilon=epsilon)
        r = run_membership_inference_attack(model, X_train, y_train, X_test, y_test)
        results[label] = r
        
        if r['attack_auc'] < 0.55:
            risk = "LOW"
            note = "attacker basically guessing randomly"
        elif r['attack_auc'] < 0.65:
            risk = "MEDIUM"
            note = "attacker has weak signal"
        else:
            risk = "HIGH"
            note = "attacker still succeeding"
            
        print(f"  loss gap: {r['loss_gap']:.3f}  |  AUC: {r['attack_auc']:.3f}  |  risk: {risk}")
        print(f"  {note}")
    
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