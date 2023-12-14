import os 
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from utils import histogram, load_svhn_dataset, visualize_class_distr, plot_images, plot_transformed_images, Standardizer, create_vectors





def train_tune_model(model, X_train, y_train, cv=5):
    '''
    train a random forest model using randomized search cross validation
    '''
    
    params = {
        'max_depth': [5,10,15, None],
        'max_features': ['sqrt', 'log2']
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=cv, verbose=True, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)

    best_score = grid_search.best_score_
    best_params = grid_search.best_estimator_.get_params()

    return grid_search.best_estimator_, best_score, best_params


def solver(model, features, train_labels, test_labels, cv, save_dir):
    '''
    this functions fits a separate random forest model to each of the 6 feature extraction methods (raw_pixel, gray_scaled, rgb_histogram, lbp, hcd, orb)
    To determine the Hyperparameters a randomized search Cross Validation is used.
    It stores the best found parameters as well as the predictions on the train and test set 
    '''
    df_preds_train = pd.DataFrame(data={'labels': train_labels})
    df_preds_test = pd.DataFrame(data={'labels': test_labels})
    best_parameters = {}

    if save_dir is not None: 
        os.makedirs(save_dir, exist_ok=True) 

    for feature in tqdm(features): 

        X_train = features[feature]['train']
        X_test = features[feature]['test']

        model, best_score, best_params = train_tune_model(model, X_train, train_labels, cv)
        
        df_preds_train[feature] = model.predict(X_train)
        df_preds_test[feature] = model.predict(X_test)

        best_parameters[feature] = {'val_acc': best_score, 'params':best_params}

    # Save the dictionary to a file
    path_params = os.path.join(save_dir, 'params.pkl')
    with open(path_params, 'wb') as file:
        pickle.dump(best_parameters, file)
      
    path_train = os.path.join(save_dir, 'train_preds.csv')
    path_test = os.path.join(save_dir, 'test_preds.csv')

    df_preds_train.to_csv(path_train)
    df_preds_test.to_csv(path_test)


def add_ensemble(df_preds):
    '''
    creates an ensemble out of all individual models. The prediction of the ensemble is based on majority voting
    '''
    df_preds['ensemble']=df_preds.iloc[:,1:].mode(axis=1).iloc[:, [0]]

    return df_preds


def compute_metrics(df_train, df_test):
    '''
    computes the accuracy for all different models on the training set and the test set
    '''
    models = df_test.columns[1:]
    df_results = pd.DataFrame(columns = ['acc_train', 'acc_test'], index = models)
    for model in models: 
        acc_test = metrics.accuracy_score(df_test['labels'], df_test[model])  
        acc_train = metrics.accuracy_score(df_train['labels'], df_train[model]) 
        df_results.loc[model] = acc_train, acc_test

    df_results = df_results.astype('float')
    return df_results