# Import important packages
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

CONFIG_PATH = "./config/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("rf_config.yaml")


def load_data():
    df = pd.read_csv(os.path.join(
        config["data_directory"], config["data_name"]))
    data = df.iloc[:, 1:31]

    X = data.loc[:, data.columns != config["target_name"]]
    y = data.loc[:, data.columns == config["target_name"]]

    number_records_fraud = len(data[data.Class == 1])
    fraud_indices = np.array(data[data.Class == 1].index)
    normal_indices = data[data.Class == 0].index
    random_normal_indices = np.random.choice(
        normal_indices, number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)
    under_sample_indices = np.concatenate(
        [fraud_indices, random_normal_indices])
    under_sample_data = data.iloc[under_sample_indices, :]

    X_undersample = under_sample_data.loc[:,
                                          under_sample_data.columns != config["target_name"]]
    y_undersample = under_sample_data.loc[:,
                                          under_sample_data.columns == config["target_name"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=42
    )


rf1 = RandomForestClassifier(
    n_estimators=config["n_estimators"],
    max_depth=config["max_depth"],
    min_samples_split=config["min_samples_split"],
    oob_score=config["oob_score"],
    random_state=config["random_state"],
    n_jobs=config["n_jobs"]
)


if __name__ == '__main__':
    load_data()
    rf1.fit(X_train, y_train)
    print(rf1.oob_score_)
    y_predprob1 = rf1.predict_proba(X_test)[:, 1]
    print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob1))
    joblib.dump(rf1, os.path.join(
        config["model_directory"], config["model_name"]))
