import pandas as pd
import sklearn
import sys

from IPython.display import display, clear_output
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score


def model(X, y, test_X, test_y, **kwargs):
    """Return predicted values, r squared, and mean squared log error."""
    if kwargs.get('hyperparams', False):
        kwargs = kwargs['hyperparams']
        regressor = RandomForestRegressor(
            n_estimators=kwargs['n_estimators'],
            criterion=kwargs['criterion'],
            max_depth=kwargs['max_depth'],
            max_features=kwargs['max_features'],
            bootstrap=kwargs['bootstrap'],
            warm_start=kwargs['warm_start'],
            ccp_alpha=kwargs['ccp_alpha']
        )
    else:
        regressor = RandomForestRegressor(**kwargs)
    regressor.fit(X, y)
    y_pred = regressor.predict(test_X)
    r2 = None if test_y is None else r2_score(y_pred, test_y)
    mse = None if test_y is None else mean_squared_log_error(y_pred, test_y)
    return y_pred, r2, mse

def validate_generation(X, y, test_X, test_y, categ,feature):
    r2divmsle = 0
    critical = None
    for value in feature:
        if categ == 'n_estimators':
            _, r2, mse = model(X, y, test_X, test_y, n_estimators=value)
        elif categ == 'criterion':
            _, r2, mse = model(X, y, test_X, test_y, criterion=value)
        elif categ == 'max_depth':
            _, r2, mse = model(X, y, test_X, test_y, max_depth=value)
        elif categ == 'max_features':
            _, r2, mse = model(X, y, test_X, test_y, max_features=value)
        elif categ == 'bootstrap':
            _, r2, mse = model(X, y, test_X, test_y, bootstrap=value)
        elif categ == 'warm_start':
            _, r2, mse = model(X, y, test_X, test_y, warm_start=value)
        elif categ == 'ccp_alpha':
            _, r2, mse = model(X, y, test_X, test_y, ccp_alpha=value)
        div = r2 / mse
        if div > r2divmsle:
            critical = value
    return critical

def validate_model(X, y, test_X, test_y):
    """This module finds out the best model generated from the data provided
    and returns the model, validated rsquared score, hyper parameters,
    and mean squared log error measure for validation data set.
    """
    print('Initiating parameter tuning')

    features = len(X.columns)
    n_estimators = [1, 10, 50, 100, 150, 300, 500]
    criterion = ['mse', 'mae']
    max_depth = [None, features, 10, 20, 30, 50, 50, 100]
    max_features = ['auto', 'sqrt', 'log2']
    bootstrap = [True, False]
    warm_start = [True, False]
    ccp_alpha = [0, 0.001, 0.01, 0.03, 0.05, 0.1, 1, 10]

    display('Tuning n_estimators... 1 / 7')
    clear_output(wait=True)
    # n_estimators
    nest = validate_generation(X, y, test_X, test_y, 'n_estimators', n_estimators)

    display('Tuning criterion...    2 / 7')
    clear_output(wait=True)
    # criterion
    crit = validate_generation(X, y, test_X, test_y, 'criterion', criterion)

    display('Tuning max_depth...    3 / 7')
    clear_output(wait=True)
    # max_depth
    max_d = validate_generation(X, y, test_X, test_y, 'max_depth', max_depth)

    display('Tuning max_features... 4 / 7')
    clear_output(wait=True)
    # max_features
    max_f = validate_generation(X, y, test_X, test_y, 'max_features', max_features)

    display('Tuning bootstrap...    5 / 7')
    clear_output(wait=True)
    # bootstrap
    boot = validate_generation(X, y, test_X, test_y, 'bootstrap', bootstrap)

    display('Tuning warm_start...   6 / 7')
    clear_output(wait=True)
    # warm_start
    ws = validate_generation(X, y, test_X, test_y, 'warm_start', warm_start)

    display('Tuning ccp_alpha...    7 / 7')
    clear_output(wait=True)
    # ccp_alpha
    ccp = validate_generation(X, y, test_X, test_y, 'ccp_alpha', ccp_alpha)

    print('Generating Model... Please wait')

    y_pred, r2, mse = model(X, y, test_X, test_y, n_estimators=nest, criterion=crit,
                            max_depth=max_d, max_features=max_f, bootstrap=boot,
                            warm_start=ws, ccp_alpha=ccp)
    print('Model Rendered')

    hypers = {
        'n_estimators': nest,
        'criterion': crit,
        'max_depth': max_d,
        'max_features': max_f,
        'bootstrap': boot,
        'warm_start': ws,
        'ccp_alpha': ccp
    }

    return pd.DataFrame({'SalePrice': y_pred}), r2, mse, hypers