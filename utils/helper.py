import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def calculate_stats_knn(data):
    X = data.drop('Species', axis=1)
    Y = data['Species'].astype('int')

    n_splits = 10

    skf = StratifiedKFold(n_splits, shuffle=True, random_state=10)
    average_accuracy=0
    for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
        X_train = X.take(train_index)
        y_train = Y.take(train_index)
        X_test = X.take(test_index)
        y_test = Y.take(test_index)

        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(X_train, y_train)

        y_pred = knn_classifier.predict(X_test)

        accuracy = round(metrics.accuracy_score(y_test, y_pred), 2)
        #print("Accuracy:", accuracy)
        average_accuracy = average_accuracy + accuracy
        cross_val_scores = cross_val_score(knn_classifier, X, Y, cv=4)
        cross_val_scores = [round(score, 2) for score in cross_val_scores]

        #print("Cross-Validation Scores:", cross_val_scores)

    average_accuracy = average_accuracy/n_splits

    return average_accuracy

def calculate_stats_mlp(data):
    X = data.drop('Species', axis=1)
    Y = data['Species'].astype('int')

    n_splits = 10

    skf = StratifiedKFold(n_splits, shuffle=True, random_state=10)
    average_accuracy = 0

    for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
        X_train = X.take(train_index)
        y_train = Y.take(train_index)
        X_test = X.take(test_index)
        y_test = Y.take(test_index)

        mlp_classifier = MLPClassifier()
        mlp_classifier.fit(X_train, y_train)

        y_pred = mlp_classifier.predict(X_test)

        accuracy = round(metrics.accuracy_score(y_test, y_pred), 2)
        average_accuracy = average_accuracy + accuracy
        cross_val_scores = cross_val_score(mlp_classifier, X, Y, cv=4)
        cross_val_scores = [round(score, 2) for score in cross_val_scores]

    average_accuracy = average_accuracy / n_splits

    return average_accuracy

def calculate_distance(source, target, data_actual):
    distance_from_source = source - data_actual
    distance_to_target = target - data_actual
    #print(abs(distance_from_source) , abs(distance_to_target))
    if abs(distance_from_source) < abs(distance_to_target):
        return 0
    else:
        return 1
    
def is_average_greater_than_half(distance_array):
    average = np.mean(distance_array)
    if average >= 0.5:
        return True
    else:
        return False