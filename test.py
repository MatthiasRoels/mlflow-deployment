"""Little script to test if the tracking server works"""
import os
import mlflow


# get port from running kubectl get services
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:30760/"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"


def main():

    features = "rooms, zipcode, median_price, school_rating, transport"
    with open("features.txt", 'w') as f:
        f.write(features)


    mlflow.set_tracking_uri("http://localhost:30496")

    mlflow.set_experiment("mlflow-testing")

    with mlflow.start_run():

        mlflow.log_params(
            {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.03,
                "min_child_weight": 50,
                "gamma": 0.001,
                "reg_lambda": 500,
            }
        )

        mlflow.log_artifact("features.txt")


if __name__ == '__main__':
    main()
