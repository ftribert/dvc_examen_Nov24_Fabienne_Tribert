
import pandas as pd 
from sklearn.model_selection import train_test_split
import os 

def Splite(Data_Raw = pd.read_csv('data/raw_data/raw.csv')):

    data_raw = Data_Raw

    # Séparation des variable cible et des Features
    df_y = data_raw[['silica_concentrate']]
    df_X = data_raw.drop('silica_concentrate', axis=1) 

    # Split en 4 DF distincts 

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

    path = "data/processed_data"
 
    if not os.path.exists(path):
        os.makedirs(path)

    # Enregistrer les DataFrames en fichiers CSV
    X_train.to_csv(os.path.join(path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(path, 'y_test.csv'), index=False)

    print ("La création des fichier train et test s'est bien éfféctuée avec une repartition de 80/20%.")

    return 

Splite()

