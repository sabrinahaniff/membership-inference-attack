import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_dataset(n=1000, seed=42):
    # synthetic medical dataset for age, bmi, blood pressure, cholesterol, glucose
    np.random.seed(seed)
    
    X = np.column_stack([
        np.random.randint(20, 80, n),       # age
        np.random.normal(27, 5, n),          # bmi
        np.random.normal(120, 15, n),        # blood pressure
        np.random.normal(200, 40, n),        # cholesterol
        np.random.normal(100, 25, n),        # glucose
    ])
    
    # diabetes risk goes up with age, bmi, and glucose levels
    # adding some noise so it's not a perfect linear relationship
    risk = (
        0.01 * X[:, 0] +
        0.05 * X[:, 1] +
        0.02 * X[:, 4] +
        np.random.normal(0, 1, n)
    )
    y = (risk > risk.mean()).astype(int)
    
    return X, y

def train_target_model(X_train, y_train, overfit=True):
    # two versions of the model:
    # overfit=True means big network, barely any regularization, memorizes training data
    # overfit=False means smaller network, stronger regularization, generalizes better
    
    # the overfit one is the more vulnerable model since it memorizes so well that
    # an attacker can tell which records it's seen before just from confidence scores
    
    if overfit:
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            alpha=0.0001,   # barely any regularization = free to memorize
            random_state=42
        )
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=100,
            alpha=0.1,      # stronger regularization = harder to memorize
            random_state=42
        )
    
    model.fit(X_train, y_train)
    return model

def get_loss(model, X, y):
    # get the loss for each individual record
    # this is the main signal the attacker uses
    # training records have way lower loss because the model has seen them before
    proba = model.predict_proba(X)
    correct_class_proba = proba[np.arange(len(y)), y]
    loss = -np.log(correct_class_proba + 1e-10)  # small epsilon to avoid log(0)
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
    
    print(f"train accuracy: {train_acc:.3f}")
    print(f"test accuracy:  {test_acc:.3f}")
    print(f"gap: {train_acc - test_acc:.3f} -- bigger gap means easier attack")
    
    train_loss = get_loss(model, X_train, y_train)
    test_loss = get_loss(model, X_test, y_test)
    
    print(f"\nmean train loss: {train_loss.mean():.3f}")
    print(f"mean test loss:  {test_loss.mean():.3f}")
    print("the loss gap is what the attacker exploits")