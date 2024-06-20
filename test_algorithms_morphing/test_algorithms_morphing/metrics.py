import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def calculate_accuracy(dataset: pd.DataFrame, algorithm: int) -> float:
    X = dataset.drop(dataset.columns[-1], axis=1)
    Y = dataset[dataset.columns[-1]].astype('int')

    n_splits = 10
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=10)
    average_accuracy = 0

    if algorithm == 1:
        classifier = RandomForestClassifier()
    elif algorithm == 2:
        classifier = MLPClassifier(max_iter=500)
    elif algorithm == 3:
        classifier = KNeighborsClassifier()
    elif algorithm == 4:
        classifier = LogisticRegression(max_iter=5000)

    for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
        X_train = X.take(train_index)
        y_train = Y.take(train_index)
        X_test = X.take(test_index)
        y_test = Y.take(test_index)
            
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = round(metrics.accuracy_score(y_test, y_pred), 2)
        average_accuracy += accuracy

    average_accuracy /= n_splits
    return average_accuracy, y_pred