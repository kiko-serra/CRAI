# CRAI

Starting dataset: iris
Target dataset: letter

### (update - 22/02/24)
Starting dataset: iris
! Old target dataset: letter -> not anymore since reducing from 20000 to 150 samples removes a lot of value from the dataset 
New target dataset: raisin (reduction only from 900 to 150 and only 2 classes)

### (update - 07/03/24)
Changing from MLPClassifier to KNeighborsClassifier for the raisin dataset. Changing this made the accuracy from 0.46 to 0.76
Change from train_test_split to StratifiedKFold

### (update - 14/03/24)
Changed from KNeighborsClassifier to LogisticRegression for the raisin dataset. Changing this made the accuracy from 0.76 to 0.95.
Instead of saving all accuracies from within the folds now it saves only the mean accuracy.
Now the column swap is done based on the index of the column instead of the name of the column.

### (update - 19/03/24)
Created notebooks to go from raisin to iris dataset.
