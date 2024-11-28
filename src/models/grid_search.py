import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

def perform_grid_search(input_dir, output_dir):
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Charger les données
    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv")

    # Définir le modèle et la grille de recherche
    model = GradientBoostingRegressor()
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    # GridSearch
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train.values.ravel())

    # Sauvegarder les meilleurs paramètres
    joblib.dump(grid_search.best_params_, f"{output_dir}/best_params.pkl")

if __name__ == "__main__":
    perform_grid_search(
        "/home/devcontainers/dvc_examen_Nov24_Fabienne_Tribert/data/processed_data",
        "/home/devcontainers/dvc_examen_Nov24_Fabienne_Tribert/models"
    )
