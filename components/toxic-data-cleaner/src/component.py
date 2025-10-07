import os
import shutil
import pandas as pd
import re
import logging
import argparse
from pathlib import Path


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"havent", " have not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


# function has to be used for both the training and the prediction data
def clean_train_data(raw_data_path, cleaned_data_path):
    # read in the data
    df_raw_data = pd.read_csv(raw_data_path)
    df_raw_data['comment_text'] = df_raw_data['comment_text'].map(lambda com: clean_text(com))
    logging.info('Cleaned text!')

    # making directories at artifacts and saving the data to artifacts
    Path(cleaned_data_path).parent.mkdir(parents=True, exist_ok=True)
    df_raw_data.to_csv('clean.csv', index=False)
    shutil.copy('clean.csv', cleaned_data_path)
    os.remove('clean.csv')


# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, help="The ingested raw data path")
    parser.add_argument('--cleaned_data_path', type=str, help="The output path for the cleaned data")
    args = parser.parse_args()
    return vars(args)


# Main execution
if __name__ == "__main__":
    clean_text('This is a sample text!')
    clean_train_data(**parse_command_line_arguments())
