import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from target_model import generate_dataset, train_target_model, get_loss

def run_membership_inference_attack(model, X_train, y_train, X_test, y_test):
    # the attacker only gets to query the model and see confidence scores
    # they don't see weights, architecture, or any training data directly
    # called a black-box attack 
    
    # get loss for training records (members) and test records (non-members)
    train_loss = get_loss(model, X_train, y_train)
    test_loss = get_loss(model, X_test, y_test)
    
    # build the attack dataset
    # features: just the loss value for each record
    # label: 1 = was in training set, 0 = was not
    attack_X = np.concatenate([train_loss, test_loss]).reshape(-1, 1)
    attack_y = np.concatenate([
        np.ones(len(train_loss)),    # training records are members
        np.zeros(len(test_loss))     # test records are non-members
    ])
    
    X_atk_train, X_atk_test, y_atk_train, y_atk_test = train_test_split(
        attack_X, attack_y, test_size=0.3, random_state=42
    )
    

    # learns: if loss is low, predict member
    attack_model = LogisticRegression()
    attack_model.fit(X_atk_train, y_atk_train)
    
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
    print("=" * 50)
    print("membership inference attack results")
    print("=" * 50)
    
    # attack the overfit model first
    print("\n[overfit model - weak regularization]")
    overfit_model = train_target_model(X_train, y_train, overfit=True)
    overfit_results = run_membership_inference_attack(
        overfit_model, X_train, y_train, X_test, y_test
    )
    print(f"  train loss: {overfit_results['train_loss_mean']:.3f}")
    print(f"  test loss:  {overfit_results['test_loss_mean']:.3f}")
    print(f"  gap: {overfit_results['loss_gap']:.3f}")
    print(f"  attack accuracy: {overfit_results['attack_accuracy']:.3f}")
    print(f"  attack AUC: {overfit_results['attack_auc']:.3f}")
    
    if overfit_results['attack_auc'] > 0.6:
        print("  --> attacker succeeded, model is leaking membership info")
    else:
        print("  --> attacker close to random guessing")
    
    # now attack the regularized model
    print("\n[regularized model - stronger regularization]")
    regular_model = train_target_model(X_train, y_train, overfit=False)
    regular_results = run_membership_inference_attack(
        regular_model, X_train, y_train, X_test, y_test
    )
    print(f"  train loss: {regular_results['train_loss_mean']:.3f}")
    print(f"  test loss:  {regular_results['test_loss_mean']:.3f}")
    print(f"  gap: {regular_results['loss_gap']:.3f}")
    print(f"  attack accuracy: {regular_results['attack_accuracy']:.3f}")
    print(f"  attack AUC: {regular_results['attack_auc']:.3f}")
    
    if regular_results['attack_auc'] > 0.6:
        print("  --> attacker still succeeding")
    else:
        print("  --> regularization helped, attacker closer to random")
    
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