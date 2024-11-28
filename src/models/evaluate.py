import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

def evaluate_model(input_dir, output_dir):
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Charger les données et le modèle
    X_test = pd.read_csv(f"{input_dir}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{input_dir}/y_test.csv")
    
    # Charger le modèle depuis le bon répertoire
    model = joblib.load(f"/home/devcontainers/dvc_examen_Nov24_Fabienne_Tribert/models/gbr_model.pkl")

    # Prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {"mse": mse, "r2": r2}

    # Sauvegarder les résultats
    pd.DataFrame({'y_test': y_test.values.ravel(), 'y_pred': y_pred}).to_csv(
        f"{input_dir}/prediction.csv", index=False
    )
    with open(f"{output_dir}/scores.json", 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    evaluate_model(
        "/home/devcontainers/dvc_examen_Nov24_Fabienne_Tribert/data/processed_data",
        "/home/devcontainers/dvc_examen_Nov24_Fabienne_Tribert/metrics"
    )

