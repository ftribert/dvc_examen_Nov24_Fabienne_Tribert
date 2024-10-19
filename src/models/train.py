import pandas as pd 
from sklearn.model_selection import GridSearchCV
import os 
import pickle

def train_model(X_train_path='../../data/processed_data/X_train_scaled.csv', 
                y_train_path='../../data/processed_data/y_train.csv',
                model_path='../../models/best_decision_tree_model.pkl'):
    
    # Chargement des données
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Chargement du meilleur modèle depuis le fichier pickle
    with open(model_path, 'rb') as file:
        best_model = pickle.load(file)

    # Entraînement du modèle avec les meilleures hyperparamètres
    best_model.fit(X_train, y_train)

    # Sauvegarde du modèle entraîné
    trained_model_filename = os.path.join(os.path.dirname(model_path), 'trained_decision_tree_model.pkl')
    with open(trained_model_filename, 'wb') as file:
        pickle.dump(best_model, file)

    return print(f"Modèle entraîné sauvegardé dans {trained_model_filename}")

train_model()