import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from target_model import generate_dataset, train_target_model, get_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_membership_inference_attack(model, X_train, y_train, X_test, y_test):
    # attacker queries the model on both training and test records
    # training records = members (label 1)
    # test records = non-members (label 0)
    
    # get loss for each record
    train_loss = get_loss(model, X_train, y_train)
    test_loss = get_loss(model, X_test, y_test)
    
    # build attack dataset
    # features: just the loss value
    # label: 1 = member, 0 = non-member
    attack_X = np.concatenate([train_loss, test_loss]).reshape(-1, 1)
    attack_y = np.concatenate([
        np.ones(len(train_loss)),   # training records = members
        np.zeros(len(test_loss))    # test records = non-members
    ])
    
    # split attack data into train/test
    X_atk_train, X_atk_test, y_atk_train, y_atk_test = train_test_split(
        attack_X, attack_y, test_size=0.3, random_state=42
    )
    
    # train attack model — simple logistic regression
    # it learns: if loss is low, predict member
    attack_model = LogisticRegression()
    attack_model.fit(X_atk_train, y_atk_train)
    
    # evaluate attack
    y_pred = attack_model.predict(X_atk_test)
    y_proba = attack_model.predict_proba(X_atk_test)[:, 1]
    
    accuracy = accuracy_score(y_atk_test, y_pred)
    auc = roc_auc_score(y_atk_test, y_proba)
    
    return {
        "attack_accuracy": accuracy,
        "attack_auc": auc,
        "train_loss_mean": train_loss.mean(),
        "test_loss_mean": test_loss.mean(),
        "loss_gap": test_loss.mean() - train_loss.mean()
    }

def compare_attack_on_models(X_train, y_train, X_test, y_test):
    # run attack on both overfit and regularized models
    # shows how regularization reduces attack success
    
    print("=" * 50)
    print("MEMBERSHIP INFERENCE ATTACK RESULTS")
    print("=" * 50)
    
    # attack overfit model
    print("\n[1] Overfit Model (weak regularization)")
    overfit_model = train_target_model(X_train, y_train, overfit=True)
    overfit_results = run_membership_inference_attack(
        overfit_model, X_train, y_train, X_test, y_test
    )
    print(f"    Train loss: {overfit_results['train_loss_mean']:.3f}")
    print(f"    Test loss:  {overfit_results['test_loss_mean']:.3f}")
    print(f"    Loss gap:   {overfit_results['loss_gap']:.3f}")
    print(f"    Attack accuracy: {overfit_results['attack_accuracy']:.3f}")
    print(f"    Attack AUC:      {overfit_results['attack_auc']:.3f}")
    print(f"    --> Attacker is {'SUCCESSFUL' if overfit_results['attack_accuracy'] > 0.6 else 'STRUGGLING'}")
    
    # attack regularized model
    print("\n[2] Regularized Model (strong regularization)")
    regular_model = train_target_model(X_train, y_train, overfit=False)
    regular_results = run_membership_inference_attack(
        regular_model, X_train, y_train, X_test, y_test
    )
    print(f"    Train loss: {regular_results['train_loss_mean']:.3f}")
    print(f"    Test loss:  {regular_results['test_loss_mean']:.3f}")
    print(f"    Loss gap:   {regular_results['loss_gap']:.3f}")
    print(f"    Attack accuracy: {regular_results['attack_accuracy']:.3f}")
    print(f"    Attack AUC:      {regular_results['attack_auc']:.3f}")
    print(f"    --> Attacker is {'SUCCESSFUL' if regular_results['attack_accuracy'] > 0.6 else 'STRUGGLING'}")
    
    return overfit_results, regular_results

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    compare_attack_on_models(X_train, y_train, X_test, y_test)