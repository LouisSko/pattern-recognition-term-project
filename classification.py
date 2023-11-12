import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


# check performance over different k's
def knn_grid_search(list_k, X_train, y_train, X_valid, y_valid):
    
    result_df = pd.DataFrame(columns=['acc_train', 'acc_valid'], index=list_k)
    result_df.index.name = 'k'
    
    for k in tqdm(list_k, 'fit k-NN with different values of k'):

        # Create a k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

        # Fit the classifier to the training data
        knn.fit(X_train, y_train)

        # Make predictions on the train data
        y_pred_train = knn.predict(X_train)

        # Make predictions on the validation data
        y_pred_valid = knn.predict(X_valid)

        acc_train = metrics.accuracy_score(y_train, y_pred_train)
        acc_valid = metrics.accuracy_score(y_valid, y_pred_valid)

        # Store results
        result_df.loc[k, :] = [acc_train, acc_valid]
    
    result_df = result_df.astype(float)
    best_k = result_df['acc_valid'].idxmax()
    
    # Display the results
    print(result_df)
    
    print(70 * '-')
    print(f'Best results obtained for k = {best_k}')
        
    return result_df, best_k


# train k-NN with the best number for k on all training data
def knn_train_test(X_train, X_test, y_train, y_test, best_k):
    # Create a k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Make predictions on the train data
    y_pred_train = knn.predict(X_train)

    # Make predictions on the test data
    y_pred_test = knn.predict(X_test)

    acc_train = metrics.accuracy_score(y_train, y_pred_train)
    acc_test = metrics.accuracy_score(y_test, y_pred_test)

    print(f'Accuracy on train set: {np.round(acc_train, 3)}')
    print(f'Accuracy on test set {np.round(acc_test, 3)}')
    
    return knn


class DataSplit:
    def __init__(self, test_size):
        self.test_size = test_size

    def get_n_splits(self, X, y=None, groups=None):
        return 1
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        train_indices, valid_indices, _, _ = train_test_split(indices, y, stratify=y, test_size=self.test_size, random_state=42)

        yield train_indices, valid_indices
        
        
def plot_error(result_df):
    # calculate the error as 1-accuracy
    sorted_df = (1-result_df).astype(float).copy()

    fig, ax1 = plt.subplots(1, 1)

    # Plot 'acc_test' and 'acc_train' with labels
    ax1.plot(sorted_df.index, sorted_df['acc_valid'], marker="o", label='Valid Error')
    ax1.plot(sorted_df.index, sorted_df['acc_train'], marker="o", label='Train Error')

    # Invert the x-axis
    ax1.invert_xaxis()

    # Add a legend
    ax1.legend()

    # Add x-axis and y-axis labels
    ax1.set_xlabel('Complexity (n_neighbors)')
    ax1.set_ylabel('Error')

    # Set the y-axis to a log scale
    ax1.set_xscale('log')

    min_index = sorted_df['acc_valid'].idxmin()
    # Add a vertical red bar at the index of the smallest entry
    ax1.axvline(x=min_index, color='red', linestyle='--', label='Smallest Entry')

    # Show the plot
    plt.show()
    

def train_tune_test(model, param_grid, X_train, y_train, X_test, y_test, cv=4):
    
    try:
        # Set up the grid search with 4-fold cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=0)

        # Fit the grid search model
        grid_search.fit(X_train, y_train)

        # Output the best parameters and the corresponding score
        best_params = grid_search.best_params_
        model = grid_search.best_estimator_
        valid_acc = grid_search.best_score_

        # Check the accuracy on the train data
        y_pred = model.predict(X_train)
        train_acc = metrics.accuracy_score(y_train, y_pred)

        # Check the accuracy on the test data
        y_pred = model.predict(X_test)
        test_acc = metrics.accuracy_score(y_test, y_pred)

        print()
        print()
        print(f'accuracy on train set: {np.round(train_acc, 3)}')
        print(f'accuracy on validation set: {np.round(valid_acc, 3)}')
        print(f'accuracy on test set: {np.round(test_acc, 3)}')
        
    except Exception as e:
        print("An error occurred:", e)
            
    return model
