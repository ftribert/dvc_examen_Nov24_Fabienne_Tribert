import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_data(input_dir, output_dir):
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Charger les données
    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")

    # Exclure la colonne 'date' et ne conserver que les colonnes numériques
    X_train = X_train.select_dtypes(include=['float64', 'int64'])
    X_test = X_test.select_dtypes(include=['float64', 'int64'])

    # Appliquer la normalisation
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sauvegarder les données normalisées
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(f"{output_dir}/X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(f"{output_dir}/X_test_scaled.csv", index=False)

if __name__ == "__main__":
    normalize_data(
        "/home/devcontainers/dvc_examen_Nov24_Fabienne_Tribert/data/processed_data",
        "/home/devcontainers/dvc_examen_Nov24_Fabienne_Tribert/data/processed_data"
    )

