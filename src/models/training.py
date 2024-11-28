import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

def train_model(input_dir, output_dir):
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Charger les données
    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv")

    # Charger les meilleurs paramètres
    best_params = joblib.load(f"{output_dir}/best_params.pkl")

    # Entraîner le modèle
    model = GradientBoostingRegressor(**best_params)
    model.fit(X_train, y_train.values.ravel())

    # Sauvegarder le modèle
    joblib.dump(model, f"{output_dir}/gbr_model.pkl")

if __name__ == "__main__":
    train_model(
        "/home/devcontainers/dvc_examen_Nov24_Fabienne_Tribert/data/processed_data",
        "/home/devcontainers/dvc_examen_Nov24_Fabienne_Tribert/models"
    )

