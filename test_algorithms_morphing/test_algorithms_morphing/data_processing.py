import pandas as pd
from sklearn.preprocessing import LabelEncoder

def remove_id_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    id_columns = [col for col in dataset.columns if 'id' == col.lower()]
    dataset = dataset.drop(id_columns, axis=1)
    return dataset

def encode_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    dataset[dataset.columns[-1]] = label_encoder.fit_transform(dataset[dataset.columns[-1]])
    return dataset

def reduce_dataset(dataset: pd.DataFrame, nr_rows: int) -> pd.DataFrame:
    target_column = dataset[dataset.columns[-1]]
    target_column.value_counts(normalize=True)
    
    target_class_percentage = {}
    for i in target_column.value_counts().index:
        target_class_percentage[i] = target_column.value_counts(normalize=True)[i], round(nr_rows * target_column.value_counts(normalize=True)[i])

    dataset = organize_dataset(dataset, target_class_percentage)
    return dataset

def organize_dataset(dataset: pd.DataFrame, target_class_percentage: dict) -> pd.DataFrame:
    ordered_data_parts = []
    for key in target_class_percentage.keys():
        class_data = dataset[dataset[dataset.columns[-1]] == key]
        num_rows = target_class_percentage[key][1]
        class_data = class_data.head(num_rows)
        ordered_data_parts.append(class_data)

    new_dataset = pd.concat(ordered_data_parts, ignore_index=True)
    return new_dataset