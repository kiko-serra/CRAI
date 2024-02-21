import pandas as pd
import math
import csv

iris = pd.read_csv("./csv/Iris.csv")
letter = pd.read_csv("./csv/letter.csv")

print(iris.head())
print(letter.head())

# 150, 6
iris_entries, iris_columns = iris.shape

# 20000, 17
letter_entries, letter_columns = letter.shape

# reorder columns of letter
letter = letter[['class', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']]

class_percentage = {}
for i in letter['class'].value_counts().index:
    class_percentage[i] = letter['class'].value_counts(normalize=True)[i], (150 * letter['class'].value_counts(normalize=True)[i]).round(0)

print(class_percentage)

final_letter_count = dict.fromkeys(letter['class'].unique(), 0)

lowest_percentage = sorted(class_percentage.items(), key=lambda x: x[1][0])[:6]

for lett in lowest_percentage:
    letter_to_remove = lett[0]
    final_letter_count[letter_to_remove] += 1


with open('./csv/final_letter.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(letter.columns)
    for index, row in letter.iterrows():
        if final_letter_count[row['class']] < class_percentage[row['class']][1]:
            writer.writerow(row)
            final_letter_count[row['class']] += 1


# Create a subset from letter with the same size as iris (150, 6)
# letter_subset = letter.sample(n=iris_entries)
# letter_subset = letter_subset.iloc[:, :6]

# letter_subset['class'].value_counts(normalize=True)
