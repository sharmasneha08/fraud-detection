import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import load_data, scale_features, split_data
from src.feature_engineering import handle_imbalance
from src.train import train_models, save_best_model
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve

if __name__ == "__main__":
    # Step 1: Load & preprocess
    df = load_data("data/raw/creditcard.csv")
    df = scale_features(df)

    # Step 2: Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 3: Handle imbalance
    X_train, y_train = handle_imbalance(X_train, y_train)

    # Step 4: Train
    models = train_models(X_train, y_train)

    # Step 5: Save best model
    best_model = save_best_model(models, X_test, y_test)

    # Step 6: Evaluate
    evaluate_model(best_model, X_test, y_test, "Best Model")
    plot_confusion_matrix(best_model, X_test, y_test, "Best Model")
    plot_roc_curve(best_model, X_test, y_test, "Best Model")

    print("\n✅ Pipeline complete!")