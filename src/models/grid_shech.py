import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import os 
import pickle

def grid_search_decision_tree(X_train_path='data/processed_data/X_train_scaled.csv', y_train_path='data/processed_data/y_train.csv'):
    # Chargement des données
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Définition de l'arbre de décision
    model = DecisionTreeRegressor()

    # Définition des paramètres à tester
    param_grid = {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'max_depth': [None, 5, 10, 15, 20] }

    # Configuration de la recherche de grille
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)

    # Exécution de la recherche de grille
    grid_search.fit(X_train, y_train)

    # Sauvegarde du meilleur modèle
    models_path = "models"
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    best_model_filename = os.path.join(models_path, 'best_decision_tree_model.pkl')
    with open(best_model_filename, 'wb') as file:
        pickle.dump(grid_search.best_estimator_, file)

    return print(f"Meilleur modèle sauvegardé dans {models_path}/best_decision_tree_model.pkl")


