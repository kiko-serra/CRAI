import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .metrics import calculate_accuracy
from .data_processing import remove_id_columns, encode_labels, reduce_dataset
from .feature_selection import get_final_dataset_dimensions, get_most_important_features
from .utils import algorithm_name, choose_algorithm

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
    
def is_average_greater_than_half(distance_array: list) -> bool:
    average = np.mean(distance_array)
    return average >= 0.5

def morphing(initial_dataset: pd.DataFrame, final_dataset: pd.DataFrame, percentage: int, nr_columns: int, nr_rows: int, algorithm_1: int, algorithm_2: int) -> None:
    dataset = initial_dataset.copy()
    dataset_delta = get_dataset_delta(initial_dataset, final_dataset, percentage, nr_columns)

    target_class_changed_index = 0
    i = 0
    accuracy_overall_1 = []
    accuracy_overall_2 = []
    meta_data = []

    with tqdm(total=int(1/percentage)) as pbar_columns:
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

            accuracy_algorithm_1, y_pred_1 = calculate_accuracy(dataset, algorithm_1)
            accuracy_algorithm_2, y_pred_2 = calculate_accuracy(dataset, algorithm_2)
            accuracy_overall_1.append(accuracy_algorithm_1)
            accuracy_overall_2.append(accuracy_algorithm_2)

            if is_average_greater_than_half(distance_array) and target_class_changed_index == 0:
                dataset.iloc[:, -1] = final_dataset.iloc[:, -1]
                target_class_changed_index = i
                print("Target feature changed to Final Dataset target at index:", i)
            if np.mean(distance_array) == 1:
                print("Converged at iteration:", i)
                break

            meta_data.append([i, np.mean(distance_array), dataset.iloc[:, -1].values[0], y_pred_1, y_pred_2, accuracy_overall_1[-1], accuracy_overall_2[-1]])
            i += 1
            pbar_columns.update(1)
    
    plt.plot(range(len(accuracy_overall_1)), accuracy_overall_1, label=algorithm_name(algorithm_1))
    plt.plot(range(len(accuracy_overall_2)), accuracy_overall_2, label=algorithm_name(algorithm_2))
    plt.axvline(x=target_class_changed_index, color='red', linestyle='--')
    plt.axhline(y=accuracy_overall_1[-1], linestyle='--')
    plt.axhline(y=accuracy_overall_2[-1], linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison ' + algorithm_name(algorithm_1) + ' vs ' + algorithm_name(algorithm_2))
    plt.legend()
    plt.show()

    meta_data = pd.DataFrame(meta_data, columns=['Iteration', 'Distance', 'Target', 'Prediction ' + algorithm_name(algorithm_1), 'Prediction ' + algorithm_name(algorithm_2), 'Accuracy ' + algorithm_name(algorithm_1), 'Accuracy ' + algorithm_name(algorithm_2)])
    print(meta_data)
    return meta_data

def analyse_datasets_accuracies(initial_dataset: pd.DataFrame, final_dataset: pd.DataFrame, percentage: int) -> None:
    algorithm_1, algorithm2 = choose_algorithm()
    initial_dataset = remove_id_columns(initial_dataset)
    final_dataset = remove_id_columns(final_dataset)

    min_rows, min_columns = get_final_dataset_dimensions(initial_dataset, final_dataset)
    initial_dataset = encode_labels(initial_dataset)
    final_dataset = encode_labels(final_dataset)

    initial_dataset_features = get_most_important_features(initial_dataset, min_columns)
    final_dataset_features = get_most_important_features(final_dataset, min_columns)

    initial_dataset = initial_dataset.iloc[:, initial_dataset_features]
    final_dataset = final_dataset.iloc[:, final_dataset_features]

    initial_dataset = reduce_dataset(initial_dataset, min_rows)
    final_dataset = reduce_dataset(final_dataset, min_rows)

    morphing(initial_dataset, final_dataset, percentage, min_columns, min_rows, algorithm_1, algorithm2)