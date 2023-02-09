import pandas as pd

training = pd.read_csv('training_dataset.csv')
training['y'] = training['w_big'].apply(lambda x: 'cluster active' if x > 1 else 'cluster idle')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(training.drop('y', axis=1), training['y'], test_size=0.2, random_state=42)

from sklearn.svm import SVC

# Create a support vector classifier
clf = SVC()

# Train the classifier
clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Predict the labels of the test set: y_pred
y_pred = clf.predict(x_test)

# Compute the confusion matrix: cm
print(confusion_matrix(y_test, y_pred))

# Print the accuracy
print(accuracy_score(y_test, y_pred))

# Print the classification report
print(classification_report(y_test, y_pred))