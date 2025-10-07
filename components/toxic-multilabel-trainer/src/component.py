import logging
import shutil
import sys
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump
from google.cloud import storage, exceptions
import pandas as pd
from pathlib import Path
import os
import argparse
import requests


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


def vectorize_data(X):
    vect = TfidfVectorizer(max_features=5000, stop_words='english')
    vect.fit(X)
    X_dtm = vect.transform(X)
    return X_dtm, vect


def save_models_to_artifact(models_path, models):
    """Saves models in models dictionary with keys as model names and values as models themselves
    Compresses the models to a tar file and saves it to the model_repo"""
    model_save_path = 'temp_path'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    for model_name, model in models.items():
        dump(model, f'{model_save_path}/{model_name}_model.joblib')

    shutil.make_archive('models', 'tar', model_save_path)
    Path(models_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.move('models.tar', models_path)
    shutil.rmtree(model_save_path)


def train_multilabel_classifier(train_path, models_path):
    models = {}
    # read in the data
    df_train_data = pd.read_csv(train_path)
    df_train_data.drop(['id'], axis=1, inplace=True)
    X = df_train_data['comment_text']
    y_all = df_train_data.drop('comment_text', axis=1)

    # vectorize the text data
    X_dtm, vectorizer = vectorize_data(X)
    logging.info('Vectorized data!')
    models['vectorizer'] = vectorizer

    cols_target = y_all.columns
    for labelnum, label in enumerate(cols_target):
        logreg = LogisticRegression(C=12.0)
        print('... Processing {}'.format(label))
        y = y_all[label]
        # train the model using X_dtm & y
        logreg.fit(X_dtm, y)
        # put model in models dict
        models[f'{labelnum}_{label}'] = logreg
        # print(f'Expected input size for {label} is {logreg.coef_.shape}')

        # make predictions with training data
        y_pred = logreg.predict(X_dtm)

        # chain current label to X_dtm
        X_dtm = add_feature(X_dtm, y)
        print('Shape of X_dtm is now {}'.format(X_dtm.shape))
        # chain current label predictions to test_X_dtm
        # test_X_dtm = add_feature(test_X_dtm, test_y)
    save_models_to_artifact(models_path, models)


# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help="Dataframe with training features")
    parser.add_argument('--models_path', type=str, help="Name of the model bucket")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    train_multilabel_classifier(**parse_command_line_arguments())
    # The *args and **kwargs is a common idiom to allow arbitrary number of arguments to functions
