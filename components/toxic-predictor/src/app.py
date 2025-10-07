import os
from flask import Flask, jsonify, request
import sys
import logging
from component import *
from requests import post

# creating the app
app = Flask(__name__)

project_id = "assignment1-402316"
model_repo = "models_de2023_group1"
clean_api_url = os.environ.get('CLEANING_API_URL')

models = load_models(project_id, model_repo)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# For testing locally
# model_repo = "/Users/huubvandevoort/Desktop/Data-Engineering/DataEngineering/development/model"
# clean_api_url = "http://127.0.0.1:5000"
# os.environ['PORT'] = "5001"
# model_names = os.listdir(model_repo)
# model_names.sort(key=lambda x: x[0])
# models = [load(os.path.join(model_repo, model_name)) for model_name in model_names if model_name.endswith('joblib')]
# project_id = '123'
# metrics_path = 'test'


# A decorator used to tell the application which URL is associated function
# the complete URL will be http://ip:port/users?name=some_value
@app.route('/', methods=["POST"])
def predict_instance():
    if request.method == 'POST':
        api_input_text = request.json.get('message')
        if not api_input_text:
            return jsonify(error="Please provide a 'message' field in the request body."), 400
        else:
            clean_response = post(url=clean_api_url, json={'text': api_input_text})
            clean_text = clean_response.json().get('clean_text', '')
            logging.info('...Received cleaned text...')
            logging.info('...Making predictions now...')
            prediction_result = predict_multilabel_classifier(clean_text, models)
            return prediction_result


@app.route('/reload_model', methods=["POST"])  # Endpoint to reload the model
def reload_model():
    global models
    models = load_models(project_id, model_repo)
    logging.info('...Reloaded models...')
    return jsonify(success=True, message="Model reloaded successfully"), 200


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
