from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from example_T12 import start
from sklearn.preprocessing import LabelEncoder

#start.analyse_datasets_accuracies(pd.read_csv('csv/Iris.csv'), pd.read_csv('csv/raisin.csv'), 10)

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
    
    # Sort the feature importances in descending order
    sorted_indices = np.argsort(rf.feature_importances_)[::-1]
    # Select the top 'nr_columns-1' indices since the target column is not included
    selected_indices = sorted_indices[:nr_columns-1]
    # Get the index of the last column (target column)
    last_column_index = len(dataset.columns) - 1
    selected_indices = np.append(selected_indices, last_column_index)
    
    return selected_indices

def organize_dataset(dataset: pd.DataFrame, target_class_percentage: dict) -> pd.DataFrame:
    ordered_data_parts = []
    for key in target_class_percentage.keys():
        # Filter the dataset for the current class
        class_data = dataset[dataset[dataset.columns[-1]] == key]
        num_rows = target_class_percentage[key][1]
        class_data = class_data.head(num_rows)
        ordered_data_parts.append(class_data)

    new_dataset = pd.concat(ordered_data_parts, ignore_index=True)
    return new_dataset

def reduce_dataset(dataset: pd.DataFrame, nr_rows: int) -> pd.DataFrame:
    target_column = dataset[dataset.columns[-1]]
    target_column.value_counts(normalize=True)
    
    target_class_percentage = {}
    for i in target_column.value_counts().index:
        target_class_percentage[i] = target_column.value_counts(normalize=True)[i], round(nr_rows * target_column.value_counts(normalize=True)[i])
    #print(target_class_percentage)
    dataset = organize_dataset(dataset, target_class_percentage)
    return dataset

def encode_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    dataset[dataset.columns[-1]] = label_encoder.fit_transform(dataset[dataset.columns[-1]])
    return dataset

#* Main function

#* Assumptions:
#* - The last column is the target column
#* - The datasets only have numerical features (except the target column which can be categorical)
#* - The datasets might have a different number of rows and columns
def analyse_datasets_accuracies(initial_dataset: pd.DataFrame, final_dataset: pd.DataFrame, percentage: int):
    min_rows, min_columns = get_final_dataset_dimensions(initial_dataset, final_dataset)
    initial_dataset = encode_labels(initial_dataset)
    final_dataset = encode_labels(final_dataset)

    initial_dataset_features = get_most_important_features(initial_dataset, min_columns)
    final_dataset_features = get_most_important_features(final_dataset, min_columns)

    #print(initial_dataset_features)
    #print(final_dataset_features)
    # Change the datasets to have the same number of columns
    initial_dataset = initial_dataset.iloc[:, initial_dataset_features]
    final_dataset = final_dataset.iloc[:, final_dataset_features]

    initial_dataset = reduce_dataset(initial_dataset, min_rows)
    final_dataset = reduce_dataset(final_dataset, min_rows)
    print("Analyse datasets accuracies")

analyse_datasets_accuracies(pd.read_csv('csv/Iris.csv'), pd.read_csv('csv/raisin.csv'), 10)