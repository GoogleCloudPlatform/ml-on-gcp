import argparse
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import scipy
import sklearn
import xgboost as xgb


from sklearn.model_selection import RandomizedSearchCV


def main(census_data_path, model_output_path, cv_iterations=1):
    TRAIN_DATA = '{}/adult.data'.format(census_data_path)
    TEST_DATA = '{}/adult.test'.format(census_data_path)

    COLUMNS = (
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'income-level'
    )

    CATEGORICAL_COLUMNS = (
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country'
    )

    with tf.gfile.Open(TRAIN_DATA, 'r') as train_data:
        train_raw_df = pd.read_csv(train_data, header=None, names=COLUMNS)

    encoders = {col:sklearn.preprocessing.LabelEncoder() for col in CATEGORICAL_COLUMNS}

    train_features_df = train_raw_df.drop('income-level', axis=1)

    for col in CATEGORICAL_COLUMNS:
        train_features_df[col] = encoders[col].fit_transform(train_features_df[col])

    train_labels_df = (train_raw_df['income-level'] == ' >50K')

    with tf.gfile.Open(TEST_DATA, 'r') as test_data:
        test_raw_df = pd.read_csv(test_data, names=COLUMNS, skiprows=1)

    test_features_df = test_raw_df.drop('income-level', axis=1)

    for col in CATEGORICAL_COLUMNS:
        test_features_df[col] = encoders[col].transform(test_features_df[col])

    test_labels_df = (test_raw_df['income-level'] == ' >50K.')

    classifier_0 = xgb.XGBClassifier(nthread=-1)

    classifier = RandomizedSearchCV(
        classifier_0,
        param_distributions={
            'max_depth': [3,4,5,6,7],
            'learning_rate': list(np.arange(0.01, 0.2, 0.01)),
            'n_estimators': scipy.stats.poisson(100),
            'gamma': scipy.stats.expon(scale=0.001),
            'reg_lambda': scipy.stats.expon(scale=0.001)
        },
        n_iter=cv_iterations
    )

    classifier.fit(train_features_df, train_labels_df, verbose=10)
    final_score = classifier.score(test_features_df, test_labels_df)

    with tf.gfile.Open(model_output_path, 'wb') as model_file:
        pickle.dump(classifier, model_file)

    return (final_score, model_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run census training job')
    parser.add_argument('--census-data-path')
    parser.add_argument('--model-output-path')
    parser.add_argument('--cv-iterations', type=int)

    args = parser.parse_args()

    print('Building income level prediction model...')

    score, output_path = main(
        args.census_data_path,
        args.model_output_path,
        args.cv_iterations
    )

    print('Model accuracy: {}'.format(score))
    print('Model path: {}'.format(output_path))

