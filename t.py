skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)

for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
        train_X = one_hot_data.take(train_index)
        train_Y = Y.take(train_index)
        test_X = one_hot_data.take(test_index)
        test_Y = Y.take(test_index)

#resto do for loop