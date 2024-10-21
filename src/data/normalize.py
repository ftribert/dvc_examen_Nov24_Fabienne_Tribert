import pandas as pd 
from sklearn.preprocessing import StandardScaler
import os 


def normalize_data(X_train_path='data/processed_data/X_train.csv', X_test_path='data/processed_data/X_test.csv'):
    # Chargement des données
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Conversion en DataFrame pour la sauvegarde
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
  
    # Définir le chemin où les fichiers CSV seront sauvegardés
    path = "data/processed_data"
 
    # Enregistrer les DataFrames normalisés en fichiers CSV
    X_train_scaled_df.to_csv(os.path.join(path, 'X_train_scaled.csv'), index=False)
    X_test_scaled_df.to_csv(os.path.join(path, 'X_test_scaled.csv'), index=False)

    return print("Normalisation des données effectuée et fichiers sauvegardés.")

normalize_data()
