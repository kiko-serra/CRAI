(note: all algorithms are using the default parameters)

iris

| Model                    | Accuracy |
|--------------------------|----------|
| KNNClassifier           | 0.97     |
| RandomForestClassifier  | 0.95     |
| LogisticRegression      | 0.97     |
| Decision Tree Classifier| 0.95     |
| Naive Bayes             | 0.96     |
| SVC                     | 0.92     |

raisin (reduction from 900 to 150 and only ['class', 'majoraxislength', 'perimeter', 'convexarea', 'area'] columns - reduction kept the distribution of the classes)
| Model                    | Accuracy |
|--------------------------|----------|
| KNNClassifier           | 0.88     |
| RandomForestClassifier  | 0.83     |
| LogisticRegression      | 0.89     |
| Decision Tree Classifier| 0.79     |
| Naive Bayes             | 0.83     |
| SVC                     | 0.79     |
