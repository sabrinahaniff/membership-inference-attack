import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_dataset(n=1000, seed=42):
    # synthetic medical dataset
    # features: age, bmi, blood_pressure, cholesterol, glucose
    # label: has_diabetes
    np.random.seed(seed)
    
    X = np.column_stack([
        np.random.randint(20, 80, n),         # age
        np.random.normal(27, 5, n),            # bmi
        np.random.normal(120, 15, n),          # blood_pressure
        np.random.normal(200, 40, n),          # cholesterol
        np.random.normal(100, 25, n),          # glucose
    ])
    
    # diabetes risk increases with age, bmi, glucose
    risk = (
        0.01 * X[:, 0] +
        0.05 * X[:, 1] +
        0.02 * X[:, 4] +
        np.random.normal(0, 1, n)
    )
    y = (risk > risk.mean()).astype(int)
    
    return X, y

def train_target_model(X_train, y_train, overfit=True):
    # deliberately overfit by using a large model with no regularization
    # this creates a gap between train and test performance
    # that gap is what the membership inference attacker exploits
    if overfit:
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            alpha=0.0001,    # very weak regularization
            random_state=42
        )
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=100,
            alpha=0.1,       # strong regularization
            random_state=42
        )
    
    model.fit(X_train, y_train)
    return model

def get_loss(model, X, y):
    # get per-sample loss for each record
    # training data will have lower loss than test data
    # this is the signal the attacker uses
    proba = model.predict_proba(X)
    correct_class_proba = proba[np.arange(len(y)), y]
    loss = -np.log(correct_class_proba + 1e-10)
    return loss

if __name__ == "__main__":
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = train_target_model(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"Generalization gap: {train_acc - test_acc:.3f}")
    print("(larger gap = easier membership inference attack)")
    
    train_loss = get_loss(model, X_train, y_train)
    test_loss = get_loss(model, X_test, y_test)
    
    print(f"\nMean train loss: {train_loss.mean():.3f}")
    print(f"Mean test loss:  {test_loss.mean():.3f}")
    print("(lower loss on training data = attacker's signal)")