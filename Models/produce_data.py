import signac
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def get_job(key):
    project = signac.get_project('dataset_per_rsquared')
    for job in project:
        if job.sp.get(key, False):
            return job

def produce_init(filename):
    """Prepare initial data for modeling."""
    training_dataset = pd.read_csv(f'../Modified Data/{filename}')
    test_dataset = pd.read_csv(f'../Data/test.csv')
    features = list(training_dataset.columns)
    features.remove('SalePrice')
    predict_feature = ['SalePrice']

    # Produce Test Data
    test_X = test_dataset.loc[:, features]
    ids_test = test_dataset.loc[:, 'Id']

    for column in features:
        if str(training_dataset.loc[:, column].dtype) == 'object':
            # Initialize encoder
            labelencoder = LabelEncoder()
            # Encode Train Data
            training_dataset.loc[:, column] = training_dataset.loc[:, column].fillna('Missing')
            training_dataset.loc[:, column] = pd.Series(labelencoder.fit_transform(training_dataset.loc[:, column]))
            # Encode Test Data
            test_X.loc[:, column] = test_X.loc[:, column].fillna('Missing')
            test_X.loc[:, column] = pd.Series(labelencoder.fit_transform(test_X.loc[:, column]))
        else:
            # Fix missing values for train data
            training_dataset.loc[:, column] = training_dataset.loc[:, column].fillna(int(training_dataset.loc[:, column].mean()))
            # Fix missing values for test data
            test_X.loc[:, column] = test_X.loc[:, column].fillna(int(test_X.loc[:, column].mean()))

    return training_dataset, test_X, ids_test


def produce_submission(ids, y, filename):
    """Produce data for submission"""
    final_submission = pd.DataFrame({'Id': ids, 'SalePrice': y})
    final_submission.to_csv(f'../Submissions/{filename}.csv', index=False)