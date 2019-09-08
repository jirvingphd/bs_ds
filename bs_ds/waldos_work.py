
# MIKE'S IMPORTS
    # from sklearn.svm import SVC
    # from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    # from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    # from sklearn.pipeline import Pipeline
    # from sklearn.decomposition import PCA
    # from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    # from scipy.stats import randint, expon
    # from sklearn.model_selection import train_test_split
    # from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.ensemble import VotingClassifier
    # from sklearn.metrics import roc_auc_score
from .bs_ds import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
# import xgboost as xbg
# from xgboost import XGBClassifier
# import time
# import re

# from .bamboo import list2df
def scale_data(data, scaler='standard', log=False):

    """Takes df or Series, scales it using desired method and returns scaled df.

    Parameters
    -----------
    data : pd.Series or pd.DataFrame
        entire dataframe of series to be scaled
    method : str
        The method for scaling to be implemented(default is 'minmax').
        Other options are 'standard' or 'robust'.
    log : bool, optional
        Takes log of data if set to True(deafault is False).

    Returns
    --------
    pd.DataFrame of scaled data.
    """

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

    scale = np.array(data)

    # reshape if needed
    if len(scale.shape) == 1:
        scale = scale.reshape(-1,1)

    # takes log if log=True
    if log == True:
        np.log(scale)


    # creates chosen scaler instance
    if scaler == 'robust':
        Scaler = RobustScaler()

    elif scaler == 'standard':
        Scaler = StandardScaler()

    elif scaler == 'minmax':
        Scaler = MinMaxScaler()
    else:
        print('No scaler specified. Defaulting to StandardScaler')
        Scaler = StandardScaler()
    scaled = Scaler.fit_transform(scale)


    # reshape and create output DataFrame
    if  scaled.shape[1] > 1:
        df_scaled = pd.DataFrame(scaled, index=data.index, columns=data.columns)

    else:
        scaled = np.squeeze(scaled)
        scaled = pd.Series(scaled)
        df_scaled = pd.DataFrame(scaled, index=data.index)

    return df_scaled


## Mike's modeling:
# import xgboost
# import sklearn
# import scipy
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
# from scipy.stats import randint, expon
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.metrics import roc_auc_score
# import xgboost as xbg
# from xgboost import XGBClassifier
# import time
# import re

def select_pca(features, n_components_list=None):

    '''
    Takes features and list of n_components to run PCA on.
    Default value of n_components_lists= None tests 2 to n_features-1.

    Params:
    ----------
    features: pd.Dataframe
    n_components_list: List of n_components (ints) to test in PCA. Default = 2:n_features-1;

    returns:
    ----------
    pd.DataFrame, displays number of components and their respective
    explained variance ratio
    '''
    import numpy as np

    # from bs_ds import list2df
    from sklearn.decomposition import PCA

    # Create list to store results in
    results = [['Model','n_components', 'Explained_Variance_ratio_']]

    # If n_components_list == None, make it 1:n-1
    if n_components_list == None:
        n_components_list = list(range(2,features.shape[1]-1))

    # Loop through list of components to do PCA on
    for n in n_components_list:

        # Creat instance of PCA class
        pca = PCA(n_components=n)
        pca.fit_transform(features)

        # Create list of n_component and Explained Variance
        component_variance = ['PCA',n, np.sum(pca.explained_variance_ratio_)]

        # Append list results list
        results.append(component_variance)

        # Use list2df to display results in DataFrame
    return list2df(results)



def train_test_dict(X, y, test_size=.25, random_state=42):

    """
    Splits data into train/test sets and returns diction with each variable its own key and value.
    """
    from sklearn.model_selection import train_test_split

    train_test = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
    train_test['X_train'] = X_train
    train_test['y_train'] = y_train
    train_test['X_test'] = X_test
    train_test['y_test'] = y_test

    return train_test

def make_estimators_dict():

    """
    Instantiates models as first step for creating pipelines.

    """
    # instantiate classifier objects
    import xgboost
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    # from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    # from sklearn.pipeline import Pipeline
    # from sklearn.decomposition import PCA
    # from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    # from scipy.stats import randint, expon
    # from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    # from sklearn.ensemble import VotingClassifier
    # from sklearn.metrics import roc_auc_score


    xgb = xgboost.XGBClassifier()
    svc = SVC()
    lr = LogisticRegression()
    gb = GradientBoostingClassifier()
    rf = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    ab = AdaBoostClassifier()

    estimators = {
        'xgb': xgb,
        'SVC': svc,
        'Logisic Regression': lr,
        'GradientBoosting': gb,
        'Random Forest': rf,
        'Decision Tree': dt,
        'AdaBoost': ab
    }
    return estimators


def make_pipes(estimators_dict, scaler=None, n_components='mle', random_state=42):

    """
    Makes pipelines for given models, outputs dictionaries with keys as names and pipeline objects as values.

    Parameters:
    ---------------
    estimators: dict,
            dictionary with name (str) as key and estimator objects as values.
    scaler: sklearn.preprocessing instance. Defaults to StandardScaler
    """
    from sklearn.pipeline import Pipeline
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()

    from sklearn.decomposition import  PCA
    # Create dictionary to store pipelines
    pipe_dict = {}

    # Instantiate piplines for each model
    for k, v in estimators_dict.items():
        pipe = Pipeline([('scaler', scaler()),
                        ('pca', PCA(n_components=n_components,random_state=random_state)),
                        ('clf', v(random_state=random_state))])
        # append to dictionary
        pipe_dict[k] = pipe

    return pipe_dict





def fit_pipes(pipes_dict, train_test, predict=True, verbose=True, score='accuracy'):

    """
    Fits piplines to training data, if predict=True, it displays a dataframe of scores.
    score can be either 'accuracy' or 'roc_auc'. rco_auc_score should be used with binary classification.

     """
    from sklearn.metrics import roc_auc_score
    import time
    fit_pipes = {}
    score_display = [['Estimator', f'Test {score}']]

    # Assert test/train sets are approriate types
    if type(train_test) == dict:
        X = train_test['X_train']
        y = train_test['y_train']
        X_test = train_test['X_test']
        y_test = train_test['y_test']

    elif type(train_test) == list:
        X = train_test[0]
        y = train_test[1]
        X_test = train_test[2]
        y_test = train_test[3]

    else:
        raise ValueError('train_test must be either list or dictionary')

    # Implement timer
    start = time.time()

    if verbose:
        print(f'fitting {len(pipes_dict)} models')

    # Fit pipes, predict if True
    for name, pipe in pipes_dict.items():

        fit_pipe = pipe.fit(X, y)
        fit_pipes['name'] = fit_pipe

        # Get accuracy or roc_auc score ,append to display list
        if predict:
            print(f'\nscoring {name} model')

            if score == 'accuracy':
                score_display.append(name, fit_pipe.score(X_test, y_test))

            elif score == 'roc_auc':
                score_display.append(name, roc_auc_score(y_test,fit_pipe.decision_function(X_test)))

            else:
                raise ValueError(f"score expected 'accuracy' of 'roc_auc', was given {score}")
    # End timer
    stop = time.time()

    if verbose:
        print(f'\nTime to fit all pipeline:{(stop-start)/60} minutes')

    # display results dataframe if prediction and verbosity
    if predict:
        display(list2df(score_display))

    return fit_pipes

#write make_config_dict, and make_random_dict
def make_config_dict(verbose=True):
    """Generates a default dictioanry of models to test and hyperparameters
    Returns dictionary of configuration to use in compare_pipes.
    Parameters:
        verbose: Defult=True, Displays contents of generated configs.

    Ex: config_dict = make_config_dict()"""
    from pprint import pprint
    import sklearn
    import xgboost
    config_dict = {
        sklearn.linear_model.LogisticRegressionCV:[{

            }],
            sklearn.linear_model.LogisticRegression:[{
                'clf__penalty':['l1'],
                'clf__C':[0.1, 1, 10, 15 ],
                'clf__tol':[1e-5, 1e-4, 1e-3],
                'clf__solver':['liblinear', 'newton-cg'],
                'clf__n_jobs':[-1]
                }, {
                'clf__penalty':['l2'],
                'clf__C':[0.1, 1, 10, 15 ],
                'clf__tol':[1e-5, 1e-4, 1e-3],
                'clf__solver':['lbfgs', 'sag'],
                'clf__n_jobs':[-1]
                }],
                sklearn.ensemble.RandomForestClassifier:[{
                    'clf__n_estimators':[10, 50, 100],
                    'clf__criterion':['gini', 'entropy'],
                    'clf__max_depth':[4, 6, 10],
                    'clf__min_samples_leaf':[0.1, 1, 5, 15],
                    'clf__min_samples_split':[0.05 ,0.1, 0.2],
                    'clf__n_jobs':[-1]
                    }],
                    sklearn.svm.SVC:[{
                        'clf__C': [0.1, 1, 10],
                        'clf__kernel': ['linear']
                        },{
                        'clf__C': [1, 10],
                        'clf__gamma': [0.001, 0.01],
                        'clf__kernel': ['rbf']
                        }],
                        sklearn.ensemble.GradientBoostingClassifier:[{
                            'clf__loss':['deviance'],
                            'clf__learning_rate': [0.1, 0.5, 1.0],
                            'clf__n_estimators': [50, 100, 150]
                            }],
                            xgboost.sklearn.XGBClassifier:[{
                                'clf__learning_rate':[.001, .01],
                                'clf__n_estimators': [1000,  100],
                                'clf__max_depth': [3, 5]
                                }]
        }
    if verbose >0:
        pprint(config_dict)
    return config_dict


# Meant for using random_pipes or compare_pipers with search = random
def make_random_config_dict(verbose=True):
    """Generates a default dictioanry of models to test and hyperparameters for a
    random grid search. Returns dictionary of configuration to use in random_pipes or compare_pipes.
    Parameters:
        verbose: Defult=True, Displays contents of generated configs.

    Ex: random_config_dict = make_random_config_dict()"""
    from pprint import pprint
    import sklearn
    import xgboost
    import scipy
    from numpy.random import randint
    random_config_dict = {
        sklearn.ensemble.RandomForestClassifier:{ # Ideal way to structure the other random searches
            'clf__n_estimators': [100 ,500, 1000],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_depth': randint(1,100),
            'clf__max_features': randint(1,100),
            'clf__min_samples_leaf': randint(1, 100),
            'clf__min_samples_split': randint(2, 10),
            'clf__n_jobs':[-1]
            },
            xgboost.sklearn.XGBClassifier:{
                'clf__silent': [False],
                'clf__max_depth': [6, 10, 15, 20],
                'clf__learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
                'clf__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'clf__colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'clf__colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'clf__min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
                'clf__gamma': [0, 0.25, 0.5, 1.0],
                'clf__reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
                'clf__n_estimators': [100]
                },
                sklearn.svm.SVC:{
                    'clf__C': scipy.stats.expon(scale=100),
                    'clf__gamma': scipy.stats.expon(scale=.1),
                    'clf__kernel': ['linear','rbf'],
                    'clf__class_weight':['balanced', None]
                    }
        }
    if verbose>0:
        pprint(random_config_dict)
    return random_config_dict



def pipe_search(estimator, params, X_train, y_train, X_test, y_test, n_components='mle',
                scaler=None, random_state=42, cv=3, verbose=2, n_jobs=-1):

    """
    Fits pipeline and performs a grid search with cross validation using with given estimator
    and parameters.

    Parameters:
    --------------
    estimator: estimator object,
            This is assumed to implement the scikit-learn estimator interface.
            Ex. sklearn.svm.SVC

    params: dict, list of dicts,
                Dictionary with parameters names (string) as keys and lists of parameter
                   settings to try as values, or a list of such dictionaries, in which case
                    the grids spanned by each dictionary in the list are explored.This enables
                    searching over any sequence of parameter settings.
                    MUST BE IN FORM: 'clf__param_'. ex. 'clf__C':[1, 10, 100]
    X_train, y_train, X_test, y_test:
            training and testing data to fit, test to model
    n_components: int, float, None or str. default='mle'
            Number of components to keep. if n_components is not set all components are kept.
            If n_components == 'mle'  Minka’s MLE is used to guess the dimension. For PCA.
    random_state: int, RandomState instance or None, optional, default=42
            Pseudo random number generator state used for random uniform sampling from lists of
            possible values instead of scipy.stats distributions. If int, random_state is the
            seed used by the random number generator; If RandomState instance, random_state
            is the random number generator; If None, the random number generator is the
            RandomState instance used by np.random.
    cv:  int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
    verbose : int,
            Controls the verbosity: the higher, the more messages.
    n_jobs : int or None, optional (default = -1)
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend
            context. -1 means using all processors. See Glossary for more details.

    Returns:
    ------------
        dictionary:
                keys are: 'test_score' , 'best_accuracy' (training validation score),
                'best_params', 'best_estimator', 'results'
    """

    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    import time

    # create dictionary to store results.
    results = {}
    # Instantiate pipeline object.
    pipe = Pipeline([('scaler', scaler),
                        ('pca', PCA(n_components=n_components,random_state=random_state)),
                        ('clf', estimator(random_state=random_state))])

    # start timer and fit pipeline.
    start = time.time()
    pipe.fit(X_train, y_train)

    # Instantiate and fit gridsearch object.
    grid = GridSearchCV(estimator = pipe,
        param_grid = params,
        scoring = 'accuracy',
        cv = cv, verbose = verbose, n_jobs=n_jobs, return_train_score = True)

    grid.fit(X_train, y_train)

    # Store results in dictionary.
    results['test_score'] = grid.score(X_test, y_test)
    results['best_accuracy'] = grid.best_score_
    results['best_params'] = grid.best_params_
    results['best_estimator'] = grid.best_estimator_
    results['results'] = grid.cv_results_

    # End timer and print results if verbosity higher than 0.
    end = time.time()
    if verbose > 0:
        name = str(estimator).split(".")[-1].split("'")[0]
        print(f'{name} \nBest Score: {grid.best_score_} \nBest Params: {grid.best_params_} ')
        print(f'\nest Estimator: {grid.best_estimator_}')
        print(f'\nTime Elapsed: {((end - start))/60} minutes')

    return results


def random_pipe(estimator, params, X_train, y_train, X_test, y_test, n_components='mle',
                scaler=None, n_iter=10, random_state=42, cv=3, verbose=2, n_jobs=-1):

    """
    Fits pipeline and performs a randomized grid search with cross validation.

    Parameters:
    --------------
    estimator: estimator object,
            This is assumed to implement the scikit-learn estimator interface.
            Ex. sklearn.svm.SVC
    params: dict,
            Dictionary with parameters names (string) as keys and distributions or
             lists of parameters to try. Distributions must provide a rvs method for
             sampling (such as those from scipy.stats.distributions).
             If a list is given, it is sampled uniformly.
            MUST BE IN FORM: 'clf__param_'. ex. 'clf__C':[1, 10, 100]
    n_components: int, float, None or str. default='mle'
            Number of components to keep. if n_components is not set all components are kept.
            If n_components == 'mle'  Minka’s MLE is used to guess the dimension. For PCA.
    X_train, y_train, X_test, y_test:
            training and testing data to fit, test to model
    scaler: sklearn.preprocessing class instance,
            MUST BE IN FORM: StandardScaler(), (default=StandardScaler())
    n_iter: int,
            Number of parameter settings that are sampled. n_iter trades off
            runtime vs quality of the solution.
    random_state: int, RandomState instance or None, optional, default=42
            Pseudo random number generator state used for random uniform sampling from lists of
            possible values instead of scipy.stats distributions. If int, random_state is the
            seed used by the random number generator; If RandomState instance, random_state
            is the random number generator; If None, the random number generator is the
            RandomState instance used by np.random.
    cv:  int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
    verbose : int,
            Controls the verbosity: the higher, the more messages.
    n_jobs : int or None, optional (default = -1)
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend
            context. -1 means using all processors. See Glossary for more details.

     Returns:
    ------------
        dictionary:
                keys are: 'test_score' , 'best_accuracy' (training validation score),
                'best_params', 'best_estimator', 'results'

    """
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.model_selection import RandomizedSearchCV
    import time

    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()

    # Start timer
    start = time.time()

    # Create dictioinary for storing results.
    results = {}
    # Instantiate Pipeline object.
    pipe = Pipeline([('scaler', scaler),
                        ('pca', PCA(n_components=n_components,random_state=random_state)),
                        ('clf', estimator(random_state=random_state))])

    # Fit pipeline to training data.
    pipe.fit(X_train, y_train)

    # Instantiate RandomizedSearchCV object.
    grid = RandomizedSearchCV(estimator = pipe,
        param_distributions = params,
        n_iter = n_iter,
        scoring = 'accuracy',
        cv = cv, verbose = verbose, n_jobs=n_jobs, return_train_score = True)

    # Fit gridsearch object to training data.
    grid.fit(X_train, y_train)

    # Store Test scores in results dictionary.
    results['test_score'] = grid.score(X_test, y_test)
    results['best_accuracy'] = grid.best_score_
    results['best_params'] = grid.best_params_
    results['best_estimator'] = grid.best_estimator_
    results['results'] = grid.cv_results_

    # End timer
    end = time.time()

    # print concise results if verbosity greater than 0.
    if verbose > 0:
        name = str(estimator).split(".")[-1].split("'")[0]
        print(f'{name} \nBest Score: {grid.best_score_} \nBest Params: {grid.best_params_} ')
        print(f'\nBest Estimator: {grid.best_estimator_}')
        print(f'\nTime Elapsed: {((end - start))/60} minutes')

    return results

def compare_pipes( X_train, y_train, X_test, y_test, config_dict=None, n_components='mle',
                 search='random',scaler=None, n_iter=5, random_state=42,
                  cv=3, verbose=2, n_jobs=-1,save_pickle=False):
    """
    Runs any number of estimators through pipeline and gridsearch(exhaustive or radomized) with cross validations,
    can print dataframe with scores, returns dictionary of all results.

    Parameters:
    --------------
    estimator: estimator object,
            This is assumed to implement the scikit-learn estimator interface.
            Ex. sklearn.svm.SVC
    params: dict, or list of dictionaries if using GridSearchcv, cannot pass lists if search='random
            Dictionary with parameters names (string) as keys and distributions or
             lists of parameters to try. Distributions must provide a rvs method for
             sampling (such as those from scipy.stats.distributions).
             If a list is given, it is sampled uniformly.
            MUST BE IN FORM: 'clf__param_'. ex. 'clf__C':[1, 10, 100]
    X_train, y_train, X_test, y_test:
            training and testing data to fit, test to model
    n_components: int, float, None or str. default='mle'
            Number of components to keep. if n_components is not set all components are kept.
            If n_components == 'mle'  Minka’s MLE is used to guess the dimension. For PCA.
    search: str, 'random' or 'grid',
            Type of gridsearch to execute, 'random' = RandomizedSearchCV,
            'grid' = GridSearchCV.
    scaler: sklearn.preprocessing class instance,
            MUST BE IN FORM: StandardScaler(), (default=StandardScaler())
    n_iter: int,
            Number of parameter settings that are sampled. n_iter trades off
            runtime vs quality of the solution.
    random_state: int, RandomState instance or None, optional, default=42
            Pseudo random number generator state used for random uniform sampling from lists of
            possible values instead of scipy.stats distributions. If int, random_state is the
            seed used by the random number generator; If RandomState instance, random_state
            is the random number generator; If None, the random number generator is the
            RandomState instance used by np.random.
    cv:  int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
    verbose : int,
            Controls the verbosity: the higher, the more messages.
    n_jobs : int or None, optional (default = -1)
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend
            context. -1 means using all processors. See Glossary for more details.

    """
    import time
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()

    if config_dict is None:

        config_dict = make_config_dict(verbose)
        print('Generating default config_dict.')


    if config_dict =="random":
        config_dict = make_random_config_dict(verbose)
        print('Generating default random_config_dict.')


    #Start timer
    begin = time.time()
    # CreateDictionary to store results from each grid search. Create list for displaying results.
    compare_dict = {}
    df_list = [['estimator', 'Test Score', 'Best Accuracy Score']]
    # Loop through dictionary instantiate pipeline and grid search on each estimator.
    for k, v in config_dict.items():

        name = str(k).split(".")[-1].split("'")[0]

        # perform RandomizedSearchCV.
        if search == 'random':

            # Assert params are in correct form, as to not raise error after running search.
            if type (v) == list:
                raise ValueError("'For random search, params must be dictionary, not list ")
            else:
                print(f"Running random_pipe for {name}...\n")
                results = random_pipe(k, v, X_train, y_train, X_test, y_test, n_components,
                                    scaler, n_iter, random_state, cv, verbose, n_jobs)
        # Perform GridSearchCV.
        elif search == 'grid':
            print(f"Running pipe_search for {name}...\n")
            results = pipe_search(k, v, X_train, y_train, X_test, y_test, n_components,
                                        scaler, random_state, cv, verbose, n_jobs )

        # Raise error if grid parameter not specified.
        else:
            raise ValueError(f"search expected 'random' or 'grid' instead got{search}")


        # append results to display list and dictionary.
        name = str(k).split(".")[-1].split("'")[0]
        print(f"Results for {name}:\n test_score: {results['test_score']}, best_accuracy: {results['best_accuracy']}.")

        compare_dict[name] = results
        df_list.append([name, results['test_score'], results['best_accuracy']])

        if save_pickle==True:
            import datetime, pickle
            currentDT = datetime.datetime.now()
            hour_adjust = int(currentDT.strftime("%H"))-4
            savetime = currentDT.strftime(f"_%m%d%y_{hour_adjust}-%M_%p")
            filename = f'pickles/comp_pipes_{name}_{savetime}.sav'
            pickle.dump(v, open(filename, 'wb'))


    # Display results if verbosity greater than 0.
    finish = time.time()
    if verbose > 0:
        print(f'\nTotal runtime: {((finish - begin)/60)}')
        display(list2df(df_list))

    return compare_dict

# from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
# from sklearn.base import clone
# import numpy as np
# from scipy import sparse
# import time
import sklearn
class MetaClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin, sklearn.base.TransformerMixin):

    """
    A model stacking classifier for sklearn classifiers. Uses Sklearn API to fit and predict,
        can be used with PipeLine and other sklearn estimators. Must be passed primary list of estimator(s)
        and secondary(meta) classifier. Secondary model trains a predicts on primary level estimators.

    Parameters:    --------------------------

    classifiers : {array-like} shape = [n_estimators]
        list of instantiated sklearn estimators.

    meta_classifier : instatiated sklearn estimator.
        This is the secondary estimator that makes the final prediction based on predicted values
        of classifiers.

    use_probability : bool, (default=False) If True calling fit will train meta_classifier on the predicted probabilities
        instead of predicted class labels.

    double_down : bool, (default=False) If True, calling fit will train meta_classifier on both the primary
        classifiers predicted lables and the original dataset. Otherwise meta_classifier will only be
        trained on primary classifier's predicted labels.

    average_probability : bool, (default = False) If True, calling fit will fit the meta_classifier with averaged
        the probabalities from primiary predictions.

    clones : bool, (default = True), If True, calling fit will fit deep copies of classifiers and meta classifier
        leaving the original estimators unmodified. False will fit the passed in classifiers directly.  This param
        is for use with non-sklearn estimators who cannot are not compatible with being cloned.  This may be unecesary
        but I read enough things about it not working to set it as an option for safe measure. It is best to clone.
    verbose : int, (0-2) Sets verbosity level for output while fitting.


    Attributes:
    --------------
    clfs_ : list, fitted classifers (primary classifiers)
    meta_clf_ : estimator, (secondary classifier)
    meta_features_ : predictions from primary classifiers

    Methods:
   ----------------
    fit(X, y, sample_weight=None): fit entire ensemble with training data, including fitting meta_classifier with meta_data
            params: (See sklearns fit model for any estimator)
                    X : {array-like}, shape = [n_samples, n_features]
                    y : {array-like}, shape =[n_samples]
                    sample_weight : array-like, shape = [n_samples], optional
    fit_transform(X, y=None, fit_params) : Refer to Sklearn docs
    predict(X) : Predict labels
    get_params(params) : get classifier parameters, refer to sklearn class docs
    set_params(params) : set classifier parameters, mostly used internally, can be used to set parameters, refer to sklearn docs.
    score(X, y, sample_weight=None): Get accuracy score
    predict_meta(X): predict meta_features, primarily used to train meta_classifier, but can be used for base ensemeble performance
    predict_probs(X) : Predict label probabilities for X.



   ***************************************** EXAMPLE*******************************************
    EXAMPLE:          # Instantiate classifier objects for base ensemble

                >>>>  xgb = XGBClassifier()
                >>>>  svc = svm.SVC()
                >>>>  gbc = GradientBoostingClassifier()

                      # Store estimators in list

                >>>>  classifiers = [xgb, svc, gbc]

                    # Instantiate meta_classifier for making final predictions

                >>>>  meta_classifier = LogisticRegression()

                    # instantiate MetaClassifer object and pass classifiers and meta_classifier
                    # Fit model with training data

                >>>>  clf = Metaclassifier(classifiers=classifiers, meta_classifier=meta_classifier)
                >>>>  clf.fit(X_train, y_train)

                    # Check accuracy scores, predict away...

                >>>>  print(f"MetaClassifier Accuracy Score: {clf.score(X_test, y_test)}")
                >>>>  clf.predict(X)
                ---------------------------------------------------------------------------

                fitting 3 classifiers...
                fitting 1/3 classifers...
                ...
                fitting meta_classifier...

                time elapsed: 6.66 minutes
                MetaClassifier Accuracy Score: 99.9   Get it!
    ***********************************************************************************************>
    """

    from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

    def __init__(self, classifiers=None, meta_classifier=None,
                 use_probability=False, double_down=False,
                 average_probs=False, clones=True, verbose=2):

        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.use_probability = use_probability
        self.double_down = double_down
        self.average_probs = average_probs
        self.clones = clones
        self.verbose = verbose



    def fit(self, X, y, sample_weight=None):

        """
        Fit base classifiers with data and meta-classifier with predicted data from base classifiers.

        Parameters:
        .--------------.-.
        X : {array-like}, shape =[n_samples, n_features]
            Training data m number of samples and number of features
        y : {array-like}, shape = [n_samples] or [n_samples, n_outputs]
            Target feature values.

        Returns:
        .--------------.-.

        self : object,
            Fitted MetaClassifier

          """
        import time
        import numpy as np
        from scipy import sparse
        from sklearn.base import clone

        start = time.time()

        # Make clones of classifiers and meta classifiers to preserve original
        if self.clones:
            self.clfs_ = clone(self.classifiers)
            self.meta_clf_ = clone(self.meta_classifier)
        else:
            self.clfs_ = self.classifiers
            self.meta_clf_ = self.meta_classifier

        if self.verbose > 0:
            print('Fitting %d classifiers' % (len(self.classifiers)))

        # Count for printing classifier count
        n = 1

        for clf in self.clfs_:

            if self.verbose > 1:
                print(f"Fitting classifier {n}/{len(self.clfs_)}")
                n +=1

            if sample_weight is None:
                clf.fit(X ,y)
            else:
                clf.fit(X, y, sample_weight)

        # Get meta_features to fit MetaClassifer
        meta_features = self.predict_meta(X)

        if self.verbose > 1:
            print("Fitting meta-classifier to meta_features")

        # Assess if X is sparse or not and stack horizontally
        elif sparse.issparse(X):
            meta_features = sparse.hstack((X, meta_features))
        else:
            meta_features = np.hstack((X, meta_features))

        # Set attribute
        self.meta_features_ = meta_features

        # Check for sample_weight and fit MetaClassifer to meta_features
        if sample_weight is None:
            self.meta_clf_.fit(meta_features, y)
        else:
            self.meta_clf_.fit(meta_features, y, sample_weight=sample_weight)

        stop = time.time()

        if self.verbose > 0:
            print(f"Estimators Fit! Time Elapsed: {(stop-start)/60} minutes")
            print("8****************************************>")

        return self



    def predict_meta(self, X):

        """
        Predicts on base estimators to get meta_features for MetaClassifier.

        Parameters:
        --------------
        X : np.array, shape=[n_samples, n_features]

        Returns:
        --------------
        meta_features : np.array, shape=[n_samples, n_classifiers]
            the 'new X' for the MetaClassifier to predict with.

        """
        import numpy as np
        # Check parameters and run approriate prediction
        if self.use_probability:

            probs = np.asarray([clf.predict_probs(X) for clf in self.clfs_])

            if self.average_probs:
                preds = np.average(probs, axis=0)

            else:
                preds = np.concatenate(probs, axis=1)

        else:
            preds = np.column_stack([clf.predict(X) for clf in self.clfs_])

        return preds

    def predict_probs(self, X):

        """
        Predict probabilities for X

        Parameters:
        --------------
        X : np.array, shape=[n_samples, n_features]

        Returns:
        --------------
        probabilities : array-like,  shape = [n_samples, n_classes]

        """
        import numpy as np
        from scipy import sparse
        meta_features = self.predict_meta(X)

        if self.double_down == False:
            return self.meta_clf_.predict_probs(meta_features)

        elif sparse.issparse(X):
            return self.meta_clf_.predict_probs(sparse.hstack((X, meta_features)))

        else:
            return self.meta_clf_.predict_probs(np.hstack((X, meta_features)))


    def predict(self, X):

        """
        Predicts target values.

        Parameters:
        --------------
        X : np.array, shape=[n_samples, n_features]

        Returns:
        --------------
        predicted labels : array-like,  shape = [n_samples] or [n_samples, n_outputs]

        """
        import numpy as np
        from scipy import sparse

        meta_features = self.predict_meta(X)

        if self.double_down == False:
            return self.meta_clf_.predict(meta_features)

        elif sparse.issparse(X):
            return self.meta_clf_.predict(sparse.hstack((X, meta_features)))

        else:
            return self.meta_clf_.predict(np.hstack((X, meta_features)))

# from sklearn.utils.estimator_checks import check_estimator



# check_estimator(MetaClassifier())
# from sklearn import svm
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
# from sklearn import tree
# import xgboost


def thick_pipe(features, target, n_components='mle',
               classifiers=[ None], test_size=.25, random_state=42, verbose=False):

    """
    Takes features and target, train/test splits and runs each through pipeline,
    outputs accuracy results models and train/test set in dictionary.

    Params:
    ------------
    features: pd.Dataframe, variable features
    target: pd.Series, classes/labels
    n_components: int, number of priniciple components, use select_pca() to determine this number
    classifiers: list, classification models put in pipeline
    test_size: float, size of test set for test_train_split (default=.25)
    split_rand: int, random_state parameter for test_train_split (default=None)
    class_rand: int, random_state parameter for classifiers (default=None)
    verbose: bool, will print pipline instances as they are created (default=False)

    Returns:
    -----------
    dictionary: keys are abbreviated name of model ('LogReg', 'DecTree', 'RandFor', 'SVC'),
    'X_train', 'X_test', 'y_train', 'y_test'. Values are dictionaries with keys for models:
    'accuracy', 'model'. values are: accuracy score,and the classification model.
     values for train/test splits. """

    from bs_ds import list2df
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    # If no classifiers specified:
    if classifiers == None:
        from sklearn.linear_model import LogisticRegression
        from sklearn import svm
        from sklearn import tree
        from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
        import xgboost

        classifiers=[
            LogisticRegression(),
            svm.SVC(),
            tree.DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            xgboost.sklearn.XGBClassifier()]

    results = [['classifier', 'score']]
    class_dict = {}

    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=test_size,
                                                        random_state=random_state)

    for classifier in classifiers:

        pipe = Pipeline([('pca', PCA(n_components=n_components,random_state=random_state)),
                         ('clf', classifier(random_state=random_state))])

        if verbose:
            print(f'{classifier}:\n{pipe}')

        pipe.fit(X_train, y_train)

        if classifier == LogisticRegression:
            name = 'LogReg'
        elif classifier == tree.DecisionTreeClassifier:
            name = 'DecTree'
        elif classifier == RandomForestClassifier:
            name = 'RandFor'
        elif classifier == AdaBoostClassifier:
            name = 'AdaBoost'
        elif classifier == GradientBoostingClassifier:
            name = 'GradBoost'
        elif classifier ==  xgboost.sklearn.XGBClassifier:
            name = 'xgb'
        else:
            name = 'SVC'

        accuracy = pipe.score(X_test, y_test)
        results.append([name, accuracy])
        class_dict[name] = {'accuracy': accuracy,'model': pipe}

    # class_dict['X_train'] = X_train
    # class_dict['X_test'] = X_test
    # class_dict['y_train'] = y_train
    # class_dict['y_test'] = y_test

    display(list2df(results))

    return class_dict


## Mike's Plotting Functions
def draw_violinplot(x , y, hue=None, data=None, title=None,
                    ticklabels=None, leg_label=None):

    '''Plots a violin plot with horizontal mean line, inner stick lines
    y must be arraylike in order to plot mean line. x can be label in data'''


    fig,ax = plt.subplots(figsize=(12,10))

    sns.violinplot(x, y, hue=hue,
                   data = data,
                   cut=2,
                   split=True,
                   scale='count',
                   scale_hue=True,
                   saturation=.7,
                   alpha=.9,
                   bw=.25,
                   palette='Dark2',
                   inner='stick'
                  ).set_title(title)

    ax.set(xlabel= x.name.title(),
          ylabel= y.name.title(),
           xticklabels=ticklabels)

    ax.axhline( y.mean(),
               label='Total Mean',
               ls=':',
               alpha=.2,
               color='xkcd:yellow')

    ax.legend().set_title(leg_label)

    plt.show()
    return fig, ax





def find_outliers(column):
    quartile_1, quartile_3 = np.percentile(column, [25, 75])
    IQR = quartile_3 - quartile_1
    low_outlier = quartile_1 - (IQR * 1.5)
    high_outlier = quartile_3 + (IQR * 1.5)
    outlier_index = column[(column < low_outlier) | (column > high_outlier)].index
    return outlier_index

# describe_outliers -- calls find_outliers
def describe_outliers(df):
    """ Returns a new_df of outliers, and % outliers each col using detect_outliers.
    """
    out_count = 0
    new_df = pd.DataFrame(columns=['total_outliers', 'percent_total'])
    for col in df.columns:
        outies = find_outliers(df[col])
        out_count += len(outies)
        new_df.loc[col] = [len(outies), round((len(outies)/len(df.index))*100, 2)]
    new_df.loc['grand_total'] = [sum(new_df['total_outliers']), sum(new_df['percent_total'])]
    return new_df

