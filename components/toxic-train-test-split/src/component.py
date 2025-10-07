import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd


def train_test_split_save(clean_data_path, train_path, test_path):
    """"
    Splits the data into training and test sets and saves them to the specified locations.
    """
    df_clean_data = pd.read_csv(clean_data_path)
    train, test = train_test_split(df_clean_data, test_size=0.2, random_state=42)
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)


# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data_path', type=str, help="Import path of the data")
    parser.add_argument('--train_path', type=str, help="Export path of the training data")
    parser.add_argument('--test_path', type=str, help="Export path of the test data")
    args = parser.parse_args()
    return vars(args)  # The vars() method returns the __dict__ (dictionary mapping) attribute of the given object.


if __name__ == '__main__':
    train_test_split_save(
        **parse_command_line_arguments())  # The *args and **kwargs is a common idiom to allow arbitrary number of
    # arguments to functions
