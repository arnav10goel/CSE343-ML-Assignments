import numpy as np
import pandas as pd
from collections import Counter

class MyDecisionTree:
    def __init__(self, max_depth=None, criterion='gini'):
        """Initialize the Decision Tree. This takes the maximum depth of the tree and the criterion to use.
        The criterion can be either "gini" for Gini Impurity or "entropy" for Information Gain and defaults
        to "gini". The tree is initialized to None and it is built using the fit function.
        
        Parameters:
            max_depth (int): The maximum depth of the tree. Default to None.
            criterion (str): The function to measure the quality of a split. Supported criteria are "gini" for Gini Impurity and "entropy" for Information Gain.
        """
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None

    def cost_function(self, groups):
        """Compute impurity of groups using the chosen criterion (Gini or Information Gain). Uses the
        formula for Gini Impurity or Information Gain depending on the criterion chosen.
        
        Parameters:
            groups (list of lists): The groups of data.
        
        Returns:
            float: The impurity value.
        """
        # Total number of instances in both groups
        n_instances = sum([len(group) for group in groups])
        impurity = 0.0
        
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            class_counts = Counter(row[-1] for row in group)
            
            for count in class_counts.values():
                prob = count / size
                if self.criterion == 'gini':
                    score += prob * prob
                elif self.criterion == 'entropy':
                    score -= prob * np.log2(prob)
                    
            # Weighted impurity score
            impurity += (1.0 - score) * (size / n_instances)
        return impurity

    def make_split(self, feature_index, threshold, dataset):
        """Split the dataset based on feature and threshold. The feature_index is the index of the feature
        to split on and the threshold is the value to split at. The dataset is split into two groups based
        on the threshold value. The left group contains all values less than the threshold and the right
        group contains all values greater than or equal to the threshold. These are stored as two separate
        lists and returned.
        
        Parameters:
            feature_index (int): The index of the feature to split on.
            threshold (float): The threshold value to split at.
            dataset (list of lists): The dataset to split.
        
        Returns:
            list: Left and right child datasets.
        """
        left, right = [], []
        for row in dataset:
            if row[feature_index] < threshold:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def get_best_split(self, dataset):
        """Get the best splitting point and resulting children for the dataset. This determines the index 
        and threshold value for the best split. The best split is the one which results in the lowest impurity.
        This uses the cost_function function to calculate the impurity of the split and 
        the make_split function to split the dataset. This is done considering all possible splits.
        The split with the lowest impurity is returned as a dictionary.
        
        Parameters:
            dataset (list of lists): The dataset to split.
        
        Returns:
            dict: The best split parameters and datasets.
        """
        best_index, best_threshold, best_score, best_groups = None, None, float("inf"), None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.make_split(index, row[index], dataset)
                impurity = self.cost_function(groups)
                if impurity < best_score:
                    best_index, best_threshold, best_score, best_groups = index, row[index], impurity, groups
        return {'index': best_index, 'threshold': best_threshold, 'groups': best_groups}

    def terminal_node(self, group):
        """Get the most common output value in a group. This is a helper function for the split function 
        which makes it easy to obtain the most common value in a group consisting of nodes which can't be
        split further and thus would form leaf nodes.
        
        Parameters:
            group (list of lists): Group of rows.
        
        Returns:
            float: The most common output value.
        """
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, depth):
        """Split a node into two children. This 
        
        Parameters:
            node (dict): The node to split.
            depth (int): The depth of the node.
        """
        left, right = node['groups']
        del(node['groups'])
        
        # If either left or right group is empty, create terminal node
        if not left or not right:
            node['left'] = node['right'] = self.terminal_node(left + right)
            return
        
        # Check for max depth
        if self.max_depth and depth >= self.max_depth:
            node['left'], node['right'] = self.terminal_node(left), self.terminal_node(right)
            return
        
        # Process left child
        node['left'] = self.get_best_split(left)
        self.split(node['left'], depth + 1)
        
        # Process right child
        node['right'] = self.get_best_split(right)
        self.split(node['right'], depth + 1)

    def fit(self, train):
        """
        Build the decision tree. This takes the training data as it is without the need for 
        separating the features and labels. You also don't need to perform any sort of encoding
        on the data for the decision tree to work.
        
        Parameters:
            train (list of lists): The training dataset.
        """

        # Print checkpoints
        print("Building the decision tree...")

        # Create the first node
        """
        The first node is the root node whose best split is determined on the entire dataset and 
        the split is made. The split function is called recursively for each node in the tree which 
        helps us build the entire tree till the leaf nodes / maximum depth is reached.
        """

        self.tree = self.get_best_split(train)
        self.split(self.tree, 1)

    def predict_row(self, node, row):
        """
        Predict the class for a single row using the decision tree. The argument node refers to the tree
        and row refers to a single row of data. This function is called recursively for each node in the
        tree. If we reach a leaf node, we simply return the output value of that node.
        
        Parameters:
            node (dict): Current node in the decision tree.
            row (list): The row of data.
        
        Returns:
            float: The predicted class.
        """
        if row[node['index']] < node['threshold']:
            if isinstance(node['left'], dict):
                return self.predict_row(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_row(node['right'], row)
            else:
                return node['right']

    def predict(self, dataset):
        """
        Predict the class for each row in the dataset. Simply calls the predict_row function for each 
        row in the dataset.
        
        Parameters:
            dataset (list of lists): The dataset to predict.
        
        Returns:
            list: List of predictions.
        """
        return [self.predict_row(self.tree, row) for row in dataset]

    def score(self, dataset):
        """Calculate the accuracy of the decision tree on a dataset. The dataset is actually 
        the testing data which is passed to the predict function to get the predictions. Then, 
        the actual labels are compared with the predicted labels to calculate the accuracy.
        
        Parameters:
            dataset (list of lists): The dataset to score.
        
        Returns:
            float: Accuracy score between 0 and 1.
        """
        actual = [row[-1] for row in dataset]
        predictions = self.predict(dataset)
        accuracy = sum([a == p for a, p in zip(actual, predictions)]) / float(len(dataset))
        return accuracy