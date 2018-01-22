# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import joblib
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import scipy
import sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


def main(mode, census_data_path, model_output_path, cv_iterations=1):
    TRAIN_DATA = os.path.join(census_data_path, 'adult.data')
    TEST_DATA = os.path.join(census_data_path, 'adult.test')

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
    train_features_df = train_raw_df.drop('income-level', axis=1)
    train_labels_df = (train_raw_df['income-level'] == ' >50K')

    with tf.gfile.Open(TEST_DATA, 'r') as test_data:
        test_raw_df = pd.read_csv(test_data, names=COLUMNS, skiprows=1)
    test_features_df = test_raw_df.drop('income-level', axis=1)
    test_labels_df = (test_raw_df['income-level'] == ' >50K.')

    if mode == 'train':
        encoders = {col:sklearn.preprocessing.LabelEncoder()
                    for col in CATEGORICAL_COLUMNS}

        for col in CATEGORICAL_COLUMNS:
            train_features_df[col] = encoders[col].fit_transform(
                train_features_df[col])

        classifier_0 = GradientBoostingClassifier()

        classifier = RandomizedSearchCV(
            classifier_0,
            param_distributions={
                'learning_rate': list(np.arange(0.01, 0.2, 0.01)),
                'max_depth': [3,4,5,6,7],
                'min_samples_split': [2,3,4,5,6],
                'min_samples_leaf': [1,2,3],
                'n_estimators': range(80, 201, 10)
            },
            n_iter=cv_iterations,
            n_jobs=-1,
            verbose=10
        )

        classifier.fit(train_features_df, train_labels_df)

        model_export = {
            'preprocessor': encoders,
            'classifier': classifier
        }

        with tf.gfile.Open(model_output_path, 'wb') as model_file:
            joblib.dump(model_export, model_file, protocol=1)
    elif mode == 'evaluate':
        with tf.gfile.Open(model_output_path, 'rb') as model_file:
            saved_model = joblib.load(model_file)
        encoders = saved_model['preprocessor']
        classifier = saved_model['classifier']
    else:
        raise ValueError('Invalid mode: {}'.format(mode))

    for col in CATEGORICAL_COLUMNS:
        test_features_df[col] = encoders[col].transform(test_features_df[col])

    final_score = classifier.score(test_features_df, test_labels_df)

    return (final_score, model_output_path, classifier.best_params_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run census training job')
    parser.add_argument('--mode',
                        choices=['train', 'evaluate'],
                        default='train')
    parser.add_argument('--census-data-path')
    parser.add_argument('--model-output-path')
    parser.add_argument('--cv-iterations', type=int, default=1)

    args = parser.parse_args()

    print('Building income level prediction model...')

    score, output_path, best_params = main(
        args.mode,
        args.census_data_path,
        args.model_output_path,
        args.cv_iterations
    )

    print('Model accuracy: {}'.format(score))
    print('Best parameters: {}'.format(best_params))
    print('Model path: {}'.format(output_path))

