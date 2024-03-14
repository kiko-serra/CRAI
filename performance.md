(note: all algorithms are using the default parameters)

iris

| Model                    | Accuracy |
|--------------------------|----------|
| KNNClassifier  | 0.97     |
| RandomForestClassifier  | 0.95     |
| LogisticRegression      | 0.97     |

raisin (reduction from 900 to 150 and only ['class', 'majoraxislength', 'perimeter', 'convexarea', 'area'] columns - reduction kept the distribution of the classes)
| Model                    | Accuracy |
|--------------------------|----------|
| KNNClassifier  | 0.88     |
| RandomForestClassifier  | 0.83     |
| LogisticRegression      | 0.89     |
