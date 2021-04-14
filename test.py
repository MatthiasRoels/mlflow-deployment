"""Little script to test if the tracking server works"""
import mlflow

mlflow.set_tracking_uri("http://localhost:32172")

mlflow.set_experiment("mlflow-testing")

mlflow.start_run()

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

mlflow.end_run()