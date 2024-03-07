# CRAI

Starting dataset: iris
Target dataset: letter

### (update - 22/02/24)
Starting dataset: iris
! Old target dataset: letter -> not anymore since reducing from 20000 to 150 samples removes a lot of value from the dataset 
New target dataset: raisin (reduction only from 900 to 150 and only 2 classes)

### (update - 07/03/24)
Changing from MLPClassifier to KNNClassifier for the raisin dataset. Changing this made the accuracy from 0.46 to 0.76
Change from train_test_split to StratifiedKFold