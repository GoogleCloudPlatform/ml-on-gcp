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
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib


def train_model(titanic_data_path, model_output_path):
    print('Loading the data...')
    try:
        with tf.gfile.Open(titanic_data_path, 'r') as data_file:
            train_df = pd.read_csv(data_file)
            print('Number of samples: {}'.format(train_df.shape[0]))

            target_name = 'Survived'
            feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

            print('Preparing the features...')
            train_features = train_df[feature_names].copy()
            train_features['Age'] = Imputer().fit_transform(train_features['Age'].values.reshape(-1, 1))
            embarked = train_features['Embarked']
            train_features['Embarked'] = embarked.fillna(embarked.mode()[0])
            train_features = pd.get_dummies(train_features)
            train_target = train_df[target_name]

            print('Training the model...')
            parameters = {'max_depth': [2, 3, 4, 5, 6, 7], 'n_estimators': [50, 100, 150, 200]}
            gsc = GridSearchCV(GradientBoostingClassifier(), parameters, n_jobs=-1, cv=5)
            gsc.fit(train_features, train_target)
            print('Best Hyper Parameters: {}'.format(gsc.best_params_))
            print('Accuracy: {}'.format(gsc.best_score_))

            with tf.gfile.Open(model_output_path, 'wb') as model_file:
                joblib.dump(gsc.best_estimator_, model_file, protocol=1)
    except Exception as e:
        print('Error: {}'.format(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train and save a model based on the Titanic dataset')
    parser.add_argument('--titanic-data-path', required=True)
    parser.add_argument('--model-output-path', required=True)

    args = parser.parse_args()
    data_path = args.titanic_data_path
    output_path = args.model_output_path
    train_model(data_path, output_path)
