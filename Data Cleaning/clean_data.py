import signac
import scipy
import pandas as pd
import pandasql as pdsql

from sklearn.preprocessing import LabelEncoder


def remove_null(dataset, percentage):
    columns = list(dataset.columns)
    null_vals = []
    for column in columns:
        null_check = f"""
            select count(*) as total
            from dataset
            where "{column}" is null
        """
        count_null = pdsql.sqldf(null_check)
        count_null['label'] = column
        null_vals.append(count_null)
    nulls = pd.concat(null_vals)
    nulls.reset_index(drop=True, inplace=True)
    order_nulls = f"""
        select label
        from nulls
        where CAST(total as float) / 1460 * 100 > {percentage}
    """
    rejected_labels = list(pdsql.sqldf(order_nulls)['label'])
    considered_labels = list(set(columns).difference(set(rejected_labels)))
    print(f"The features for which null values were greater than {percentage}% are "
          f"{', '.join(rejected_labels)}")
    return dataset[considered_labels]


def analyse_rsquared(dataset):
    """Return R-squared metric per column"""
    def rsquared(x, Y):
        """ Return R^2 metric by performing linear regression"""
        if str(x.dtype) == 'object':
            # Fix missing values
            x = x.fillna('Missing')
            # Perform encoding
            labelencoder = LabelEncoder()
            x = labelencoder.fit_transform(x)
            # x = pd.Series(x)
        else:
            # Fix missing values
            x = x.fillna(int(x.mean()))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, Y)
        return (r_value ** 2) * 100

    rsquared_testing = {'label': [], 'variance_explained': []}

    for column in dataset:
        query = f"""
            select "{column}" as label, SalePrice
            from dataset
        """
        df = pdsql.sqldf(query)
        rs_metric = rsquared(df['label'], df['SalePrice'])
        rsquared_testing['label'].append(column)
        rsquared_testing['variance_explained'].append(rs_metric)

    final_rs_df = pd.DataFrame(rsquared_testing)
    final_rs_df = final_rs_df.sort_values(by='variance_explained', ascending=False)
    final_rs_df.reset_index(drop=True, inplace=True)
    return final_rs_df

def store_selected_rsquared_limit(dataset, features, limit, filename):
    query = f"""
        select label
        from features
        where variance_explained > {limit}
    """
    final_data = dataset[list(pdsql.sqldf(query)['label'])]
    final_data.to_csv(f'../Modified Data/{filename}.csv', index=False)
    return final_data
