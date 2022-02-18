import os
import pickle

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import mlflow
from mlflow.sklearn import log_model

# get port from running kubectl get services
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://10.99.12.53:9000/"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"


def main():

    mlflow.set_tracking_uri("http://10.100.163.22:5000")
    mlflow.set_experiment("test-iris")

    X, y = load_iris(return_X_y=True)

    with mlflow.start_run():

        #clf = LogisticRegression(random_state=0)
        clf = LinearSVC(random_state=0)
        clf.fit(X, y)

        mlflow.log_params(
            clf.get_params()
        )

        mlflow.log_metric("accuracy - train", clf.score(X, y))

        mlflow.sklearn.log_model(clf, "iris_svm")


if __name__ == '__main__':
    main()
