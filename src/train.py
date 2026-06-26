import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
        print(f"{name} - CV F1 Score: {scores.mean():.4f}")
        trained[name] = model
    return trained

def save_best_model(models, X_test, y_test):
    best_name, best_model, best_score = None, None, 0
    for name, model in models.items():
        score = f1_score(y_test, model.predict(X_test))
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model
    joblib.dump(best_model, "models/fraud_model.pkl")
    print(f"Best Model: {best_name} with F1: {best_score:.4f}")
    print("Model saved to models/fraud_model.pkl")
    return best_model