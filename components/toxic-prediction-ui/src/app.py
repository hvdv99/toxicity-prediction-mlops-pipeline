from flask import Flask, request, render_template
from requests import post
import os
import logging
import sys

predictor_api_url = os.environ.get('PREDICTOR_API_URL')

app = Flask(__name__)


def clean_prediction_response(prediction_response):
    for key in prediction_response.keys():
        val = prediction_response.get(key, 999)
        val = format(float(val) * 100, '.1f')
        prediction_response[key] = round(float(val), 1)
    return prediction_response


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("input-form-page.html")
        # Do not write /templates/input-form-page.html because flask already knows its in templates

    if request.method == 'POST':
        input_text = request.form.get('message', '')
        if input_text:
            prediction_response = post(url=predictor_api_url, json={'message': input_text})
            if prediction_response.status_code == 200:
                cleaned_prediction_response = clean_prediction_response(prediction_response.json())
                return render_template("response-page.html",
                                       cleaned_prediction_response=cleaned_prediction_response)
            else:
                logging.info("There is something wrong with the request")
        # Do not write /templates/response-page.html because flask already knows its in templates


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
