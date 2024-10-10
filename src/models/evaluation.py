import pandas as pd 
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

def evaluation(X_test_path='../../data/processed_data/X_test_scaled.csv', 
               y_test_path='../../data/processed_data/y_test.csv',
               model_path='../../models/trained_decision_tree_model.pkl', 
               predictions_path='../../data/predictions.csv', 
               metrics_path='../../metrics/scores.json'):
    
    # Chargement des données
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    # Chargement du modèle entraîné
    with open(model_path, 'rb') as file:
        trained_model = pickle.load(file)

    # Prédictions
    predictions = trained_model.predict(X_test)

    # Calcul des métriques d'évaluation
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Sauvegarde des prédictions dans un DataFrame
    predictions_df = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': predictions})
    predictions_df.to_csv(predictions_path, index=False)

    # Sauvegarde des métriques dans un fichier JSON
    metrics = {'MSE': mse, 'R2': r2}
    with open(metrics_path, 'w') as json_file:
        json.dump(metrics, json_file)

    print(f"Prédictions sauvegardées dans {predictions_path}")
    return    print(f"Métriques d'évaluation sauvegardées dans {metrics_path}")


evaluation()