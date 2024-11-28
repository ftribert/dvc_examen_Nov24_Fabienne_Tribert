import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_path, output_dir):
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Charger les données
    data = pd.read_csv(input_path)

    # Exclure la colonne 'date' et définir la variable cible 'silica_concentrate'
    X = data.drop(columns=['date', 'silica_concentrate'])  # Variables explicatives
    y = data['silica_concentrate']  # Variable cible

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sauvegarder les ensembles
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

if __name__ == "__main__":
    split_data(
        "/home/devcontainers/dvc_examen_Nov24_Fabienne_Tribert/data/raw/raw.csv",
        "/home/devcontainers/dvc_examen_Nov24_Fabienne_Tribert/data/processed_data"
    )


