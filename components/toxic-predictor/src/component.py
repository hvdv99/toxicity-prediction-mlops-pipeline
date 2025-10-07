import logging
import shutil
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from joblib import load
from google.cloud import storage
import pandas as pd
from pathlib import Path
import json
import os
import argparse
import numpy as np


# create a function to add features
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


def load_models_from_artifact(model_path, model_names):
    trained_models = []
    # unpack the tar file
    temp_path = 'temp_path'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    shutil.unpack_archive(model_path, temp_path, 'tar')

    # loop over files from model_names and load the models
    for model_name in model_names:
        model = load(f'{temp_path}/{model_name}')
        trained_models.append(model)
    # remove the temp_path
    shutil.rmtree(temp_path)
    return trained_models


def load_models_from_gcs(project_id, model_repo, model_names):
    trained_models = []
    temp_path = 'temp_path'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(model_repo)
    for model_name in model_names:
        local_file = os.path.join(temp_path, model_name)
        blob = bucket.blob(model_name)
        blob.download_to_filename(local_file)
        model = load(local_file)
        trained_models.append(model)
        os.remove(local_file)
    return trained_models


def load_models(project_id, model_repo, validation_data=False):
    model_names = ['0_toxic_model.joblib',
                   '1_severe_toxic_model.joblib',
                   '2_obscene_model.joblib',
                   '3_threat_model.joblib',
                   '4_insult_model.joblib',
                   '5_identity_hate_model.joblib',
                   'vectorizer_model.joblib']
    if validation_data:
        trained_models = load_models_from_artifact(model_path=model_repo, model_names=model_names)
    else:
        trained_models = load_models_from_gcs(project_id, model_repo, model_names)
    return trained_models


def save_predictions_to_artifact(y, y_pred, y_pred_proba, predicted_path):
    # temp path
    temp_path = 'temp_path'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    # export y_pred to a csv file
    y_pred = pd.DataFrame(y_pred, columns=y.columns)
    y_pred.to_csv(f'{temp_path}/y_pred.csv', index=False)
    # export y_pred_proba to a csv file
    y_pred_proba = pd.DataFrame(y_pred_proba, columns=y.columns)
    y_pred_proba.to_csv(f'{temp_path}/y_pred_proba.csv', index=False)
    # export y to a csv file
    y.to_csv(f'{temp_path}/y.csv', index=False)
    # make a tar file
    shutil.make_archive('metrics', 'tar', temp_path)
    # save the tar file to the metrics_path
    Path(predicted_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.move('metrics.tar', predicted_path)
    # remove the temp_path
    shutil.rmtree(temp_path)


def predict_multilabel_classifier(predict_data, models,
                                  predicted_path=None, validation_data=False):
    """Takes models from model repo and makes predictions on the predict data.
    If validation data is True, then it also performs validation on the predict data.
    """

    if validation_data:
        # read in the data
        df_predict_data = pd.read_csv(predict_data)
        df_predict_data.drop(['id'], axis=1, inplace=True)
        y_all = df_predict_data.drop('comment_text', axis=1)
        X = df_predict_data['comment_text']
    else:
        X = [predict_data]

    # load the vectorizer
    vectorizer = models[-1]
    X_dtm = vectorizer.transform(X)
    logging.info('Vectorized data!')

    # load the models
    toxic_models = models[:-1]
    # make predictions with predict data
    y_pred = []
    y_pred_proba = []
    for model in toxic_models:
        y_pred.append(model.predict(X_dtm))
        y_pred_proba.append(model.predict_proba(X_dtm)[:, -1])
        X_dtm = add_feature(X_dtm, y_pred[-1])
    # if validation data is True, then compute the evalution metrics for the predict data
    if validation_data:
        y_pred = np.array(y_pred).T
        y_pred_proba = np.array(y_pred_proba).T
        save_predictions_to_artifact(y_all, y_pred, y_pred_proba, predicted_path)
        # macro_metrics = calculate_metrics_and_save(y_all, y_pred, 'macro', metrics_path)
    else:
        labels = ['toxic',
                  'severe_toxic',
                  'obscene',
                  'threat',
                  'insult',
                  'identity_hate']
        instance_predictions = {}
        for label, predict_p in zip(labels, y_pred_proba):
            instance_predictions[label] = format(float(predict_p), '.3f')
            # need to cast to non-scientific float since predict_p is np.array

        return instance_predictions


# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, help="GCP project id")
    parser.add_argument('--predict_data', type=str, help="Dataframe with training features")
    parser.add_argument('--model_repo', type=str, help="Name of the model bucket")
    parser.add_argument('--predicted_data_path', type=str, help="Path to be used for saving predicted data")
    parser.add_argument('--validation_data', action='store_true',
                        help="Weather to perform validation on the data. Now this is just a flag (no value needed). "
                             "If not given, it sets arguments to False")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_command_line_arguments()
    models = load_models(project_id=args.get('project_id'),
                         model_repo=args.get('model_repo'),
                         validation_data=args.get('validation_data')
                         )
    predict_multilabel_classifier(predict_data=args.get('predict_data'),
                                  models=models,
                                  predicted_path=args.get('predicted_data_path'),
                                  validation_data=args.get('validation_data')
                                  )
    # The *args and **kwargs is a common idiom to allow arbitrary number of arguments to functions
