from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from example_T12 import start
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

#start.analyse_datasets_accuracies(pd.read_csv('csv/Iris.csv'), pd.read_csv('csv/raisin.csv'), 10)
def remove_id_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    id_columns = [col for col in dataset.columns if 'id' == col.lower()]
    dataset = dataset.drop(id_columns, axis=1)
    return dataset

def get_final_dataset_dimensions(initial_dataset: pd.DataFrame, final_dataset: pd.DataFrame) -> tuple:
    di_rows, di_columns = initial_dataset.shape
    df_rows, df_columns = final_dataset.shape
    # Return the minimum number of rows and columns. Remove 1 from the columns since the target column is not included
    return min(di_rows, df_rows), min(di_columns, df_columns)-1

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

    dataset = organize_dataset(dataset, target_class_percentage)
    return dataset

def encode_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    dataset[dataset.columns[-1]] = label_encoder.fit_transform(dataset[dataset.columns[-1]])
    return dataset

def get_dataset_delta(initial_dataset: pd.DataFrame, final_dataset: pd.DataFrame, percentage: int, nr_columns: int) -> pd.DataFrame:
    new_dataset = pd.DataFrame()
    for i in range(nr_columns):
        new_dataset[i] = (abs(final_dataset.iloc[:, i]) - abs(initial_dataset.iloc[:, i])) * percentage

    return new_dataset

def calculate_dataset_distance(initial_value: int, final_value: int, current_value: int) -> int:
    distance_from_initial = initial_value - current_value
    distance_from_final = final_value - current_value
    total_distance = final_value - initial_value
    if distance_from_initial == 0:
        return -1
    elif distance_from_final == 0:
        return 1
    else:
        return (current_value - initial_value) / total_distance
    
def calculate_accuracy_rfc(dataset: pd.DataFrame) -> float:
    X = dataset.drop(dataset.columns[-1], axis=1)
    Y = dataset[dataset.columns[-1]].astype('int')

    n_splits = 10
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=10)
    average_accuracy=0

    for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
        X_train = X.take(train_index)
        y_train = Y.take(train_index)
        X_test = X.take(test_index)
        y_test = Y.take(test_index)

        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)

        accuracy = round(metrics.accuracy_score(y_test, y_pred), 2)
        average_accuracy = average_accuracy + accuracy

    average_accuracy = average_accuracy/n_splits

    return average_accuracy

def calculate_accuracy_mlp(dataset: pd.DataFrame) -> float:
    X = dataset.drop(dataset.columns[-1], axis=1)
    Y = dataset[dataset.columns[-1]].astype('int')

    n_splits = 10
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=10)
    average_accuracy=0

    for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
        X_train = X.take(train_index)
        y_train = Y.take(train_index)
        X_test = X.take(test_index)
        y_test = Y.take(test_index)

        mlp_classifier = MLPClassifier(max_iter=500)
        mlp_classifier.fit(X_train, y_train)

        y_pred = mlp_classifier.predict(X_test)

        accuracy = round(metrics.accuracy_score(y_test, y_pred), 2)
        average_accuracy = average_accuracy + accuracy

    average_accuracy = average_accuracy/n_splits

    return average_accuracy

def is_average_greater_than_half(distance_array: list) -> bool:
    average = np.mean(distance_array)
    if average >= 0.5:
        return True
    else:
        return False    

def morphing(initial_dataset: pd.DataFrame, final_dataset: pd.DataFrame, percentage: int, nr_columns: int, nr_rows: int) -> None:
    dataset = initial_dataset.copy()
    dataset_delta = get_dataset_delta(initial_dataset, final_dataset, percentage, nr_columns)

    target_class_changed_index = 0
    i = 0
    accuracy_overall_rfc = []
    accuracy_overall_mlp = []
    while True:
        distance_array = []
        for column in range(nr_columns-1):
            for row in range(nr_rows):
                final_value = final_dataset.iloc[row, column]
                current_value = dataset.iloc[row, column]
                delta_value = dataset_delta.iloc[row, column]
                distance = final_value - current_value
                if distance > 0:
                    if current_value + delta_value > final_value:
                        dataset.iloc[row, column] = final_value
                    else:
                        dataset.iloc[row, column] += delta_value
                elif distance < 0:
                    if current_value - delta_value < final_value:
                        dataset.iloc[row, column] = final_value
                    else:
                        dataset.iloc[row, column] -= delta_value

                dataset_distance = calculate_dataset_distance(initial_dataset.iloc[row, column], final_dataset.iloc[row, column], dataset.iloc[row, column])
                distance_array.append(dataset_distance)

        accuracy_overall_rfc.append(calculate_accuracy_rfc(dataset))
        accuracy_overall_mlp.append(calculate_accuracy_mlp(dataset))

        if is_average_greater_than_half(distance_array) and target_class_changed_index == 0:
            dataset.iloc[:, -1] = final_dataset.iloc[:, -1]
            target_class_changed_index = i
            print("Target feature changed to Final Dataset target at index:", i)
        #print("Iteration:", i, "Distance avg", np.mean(distance_array) )
        if np.mean(distance_array) == 1:
            print("Converged at iteration:", i)
            break
        i = i + 1
    plt.plot(range(len(accuracy_overall_rfc)), accuracy_overall_rfc, label='RFC')
    plt.plot(range(len(accuracy_overall_mlp)), accuracy_overall_mlp, label='MLP')
    plt.axvline(x=target_class_changed_index, color='red', linestyle='--')
    plt.axhline(y=accuracy_overall_rfc[-1], linestyle='--')
    plt.axhline(y=accuracy_overall_mlp[-1], linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison KNN vs RFC')
    plt.legend()
    plt.show()
#* Main function

#* Assumptions:
#* - The last column is the target column
#* - The datasets only have numerical features (except the target column which can be categorical)
#* - The datasets might have a different number of rows and columns
def analyse_datasets_accuracies(initial_dataset: pd.DataFrame, final_dataset: pd.DataFrame, percentage: int) -> None:
    initial_dataset = remove_id_columns(initial_dataset)
    final_dataset = remove_id_columns(final_dataset)

    min_rows, min_columns = get_final_dataset_dimensions(initial_dataset, final_dataset)
    initial_dataset = encode_labels(initial_dataset)
    final_dataset = encode_labels(final_dataset)

    initial_dataset_features = get_most_important_features(initial_dataset, min_columns)
    final_dataset_features = get_most_important_features(final_dataset, min_columns)

    # Change the datasets to have the same number of columns
    initial_dataset = initial_dataset.iloc[:, initial_dataset_features]
    final_dataset = final_dataset.iloc[:, final_dataset_features]

    initial_dataset = reduce_dataset(initial_dataset, min_rows)
    final_dataset = reduce_dataset(final_dataset, min_rows)

    morphing(initial_dataset, final_dataset, percentage, min_columns, min_rows)

analyse_datasets_accuracies(pd.read_csv('csv/Iris.csv'), pd.read_csv('csv/raisin.csv'), 0.01)