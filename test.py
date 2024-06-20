import pandas as pd
from test_algorithms_morphing import analyse_datasets_accuracies

initial_dataset = pd.read_csv('csv/Iris.csv')
final_dataset = pd.read_csv('csv/raisin.csv')
percentage = 0.01

analyse_datasets_accuracies(initial_dataset, final_dataset, percentage)


from pymfe.mfe import MFE
x = X.values
y = Y.values
mfe = MFE()
try:
    mfe.fit(x,y)
    ft = mfe.extract()
except:
    print("Error in extracting Metafeatures")
    ft = "Error in extracting Metafeatures"