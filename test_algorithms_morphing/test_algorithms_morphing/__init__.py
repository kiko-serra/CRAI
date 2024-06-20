from .data_processing import remove_id_columns, encode_labels, reduce_dataset
from .feature_selection import get_final_dataset_dimensions, get_most_important_features
from .metrics import calculate_accuracy
from .morphing import morphing, analyse_datasets_accuracies
from .utils import algorithm_name, choose_algorithm