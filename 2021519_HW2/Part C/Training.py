import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
from MyDecisionTree import MyDecisionTree

# Open the dataset
pd = pd.read_csv("Thyroid_data.csv")

# Separate the features and the labels
X = pd.iloc[:, :-1].values
y = pd.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Join the features and labels for training
train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)

# Join the features and labels for testing
test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)

# Take Input for max_depth and criterion
max_depth = input("Enter the maximum depth of the tree: ")
criterion = input("Enter the criterion to use for splitting: ")

if max_depth.isdigit():
    max_depth = int(max_depth)
else:
    print("Invalid max_depth entered. Using default max_depth None.")
    max_depth = None

if criterion not in ["gini", "entropy"]:
    print("Invalid criterion entered. Using default criterion gini.")
    criterion = "gini"

# Create the model
if criterion and max_depth:
    model = MyDecisionTree(max_depth=max_depth, criterion=criterion)
elif criterion:
    model = MyDecisionTree(criterion=criterion)
elif max_depth:
    model = MyDecisionTree(max_depth=max_depth)
else:
    model = MyDecisionTree()

print("Training the model...")

# Fit the model
model.fit(train_data)

print("Model trained!")

# Dump the model in a pickle file
with open("MyDecisionTree.pkl", "wb") as f:
    pickle.dump(model, f)

# Make predictions
predictions = model.predict(test_data)

# Calculate the accuracy
accuracy = model.score(test_data)
print("Testing Data Accuracy:", accuracy)

# Make predictions on the training data
predictions_train = model.predict(train_data)

# Calculate the accuracy
print("\nTraining Data Accuracy:", model.score(train_data))