from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

def get_final_dataset_dimensions(initial_dataset: pd.DataFrame, final_dataset: pd.DataFrame) -> tuple:
    di_rows, di_columns = initial_dataset.shape
    df_rows, df_columns = final_dataset.shape
    return min(di_rows, df_rows), min(di_columns, df_columns)

def get_most_important_features(dataset: pd.DataFrame, nr_columns: int) -> list:
    # Assuming the last column is the target column
    X = dataset.drop(dataset.columns[-1], axis=1)
    Y = dataset[dataset.columns[-1]]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75)
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    
    sorted_indices = np.argsort(rf.feature_importances_)[::-1]
    selected_indices = sorted_indices[:nr_columns]
    # ?test code
    print("Most important features:")
    
    return selected_indices
# main function
def analyse_datasets_accuracies(initial_dataset: pd.DataFrame, final_dataset: pd.DataFrame, percentage: int):
    smaller_rows, smaller_columns = get_final_dataset_dimensions(initial_dataset, final_dataset)
    initial_dataset_features = get_most_important_features(initial_dataset, smaller_columns)
    final_dataset_features = get_most_important_features(final_dataset, smaller_columns)
    print("Analyse datasets accuracies")


analyse_datasets_accuracies(pd.read_csv("../../../csv/Iris.csv"), pd.read_csv("../../../csv/raisin.csv"), 10)