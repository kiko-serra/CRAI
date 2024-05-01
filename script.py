import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.helper import calculate_stats_knn, calculate_stats_mlp, calculate_distance, is_average_greater_than_half

iris = pd.read_csv('./final_csv/reduced_iris.csv')
raisin = pd.read_csv('./final_csv/reduced_raisin.csv')

num_columns = iris.shape[1]
# print("Number of columns:", num_columns)
iris = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']]
raisin = raisin[['majoraxislength', 'perimeter', 'convexarea', 'area', 'class']]
iris['Species'] = iris['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

iris_columns = iris.columns
raisin_columns = raisin.columns

accuracy_overall_knn = []
accuracy_overall_mlp = []
data = iris.copy()

difference1 = pd.DataFrame()

for i in range(num_columns-1):
    difference1[i] = (abs(raisin.iloc[:, i]) - abs(iris.iloc[:, i])) * 0.01

print(difference1.head())
print(iris.head())
print(raisin.head())

percentage = 0.25
class_changed = 0
i = 0
while True:
    distance_array = []
    for index in range(num_columns-1):
        for second_index in range(150):
            #print("Index:", index, "Second Index:", second_index)
            delta = raisin.iloc[second_index, index] - data.iloc[second_index, index]
            if delta > 0:
                if data.iloc[second_index, index] + difference1.iloc[second_index, index] > raisin.iloc[second_index, index]:
                    data.iloc[second_index, index] = raisin.iloc[second_index, index]
                else:
                    data.iloc[second_index, index] = data.iloc[second_index, index] + difference1.iloc[second_index, index]
            elif delta < 0:
                if data.iloc[second_index, index] - difference1.iloc[second_index, index] < raisin.iloc[second_index, index]:
                    data.iloc[second_index, index] = raisin.iloc[second_index, index]
                else:
                    data.iloc[second_index, index] = data.iloc[second_index, index] - difference1.iloc[second_index, index]
            
            distance = calculate_distance(iris.iloc[second_index, index], raisin.iloc[second_index, index], data.iloc[second_index, index])
            distance_array.append(distance)

    accuracy_overall_knn.append(calculate_stats_knn(data))
    accuracy_overall_mlp.append(calculate_stats_mlp(data))

    if is_average_greater_than_half(distance_array) and class_changed == 0:
        data.iloc[:, -1] = raisin.iloc[:, -1]
        class_changed = i
        print("Target feature changed to Raisin at index:", i)
    print("Iteration:", i, "Distance avg", np.mean(distance_array) )
    if np.mean(distance_array) == 1:
        print("Converged at iteration:", i)
        break
    i = i + 1

plt.plot(accuracy_overall_knn)
plt.axvline(x=class_changed, color='red', linestyle='--')
plt.axhline(y=accuracy_overall_knn[-1], color='green', linestyle='--')
plt.xlabel('KNeighborsClassifier - Iris -> Raisin')
plt.ylabel('Accuracy')
plt.title('Accuracy Overall')
plt.show()

plt.plot(accuracy_overall_mlp)
plt.axvline(x=class_changed, color='red', linestyle='--')
plt.axhline(y=accuracy_overall_mlp[-1], color='green', linestyle='--')
plt.xlabel('MLPClassifier - Iris -> Raisin')
plt.ylabel('Accuracy')
plt.title('Accuracy Overall')
plt.show()