import argparse
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent

NUMERIC = [
    "Age", "MonthlyIncome", "NumberOfTrips",
    "NumberOfPersonVisiting", "NumberOfChildrenVisiting",
]
CATEGORICAL = [
    "TypeofContact", "CityTier", "Occupation", "Gender",
    "PreferredPropertyStar", "MaritalStatus", "Designation",
]
BINARY = ["Passport", "OwnCar"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=PROJECT_DIR / "data" / "processed",
    )
    args = parser.parse_args()

    config_path = PROJECT_DIR / "deployment" / "model_config.json"
    config = json.loads(config_path.read_text())

    # Require the versions used to train the evaluated model.
    for package in ["scikit-learn", "pandas", "numpy", "scipy", "joblib"]:
        expected = config["package_versions"][package]
        actual = version(package)
        if actual != expected:
            raise RuntimeError(
                f"{package}: expected {expected}, found {actual}"
            )

    features = config["feature_columns"]
    threshold = float(config["classification_threshold"])

    X_train = pd.read_csv(args.data_dir / "X_train.csv")
    X_test = pd.read_csv(args.data_dir / "X_test.csv")
    y_train = pd.read_csv(args.data_dir / "y_train.csv")["ProdTaken"]
    y_test = pd.read_csv(args.data_dir / "y_test.csv")["ProdTaken"]
    train_groups = pd.read_csv(
        args.data_dir / "groups_train.csv"
    )["profile_group"]
    test_groups = pd.read_csv(
        args.data_dir / "groups_test.csv"
    )["profile_group"]

    if list(X_train.columns) != features or list(X_test.columns) != features:
        raise ValueError("Input columns or their order differ from configuration.")

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts differ.")

    if len(train_groups) != len(X_train) or len(test_groups) != len(X_test):
        raise ValueError("Profile group row counts differ.")

    if set(train_groups) & set(test_groups):
        raise ValueError("Training and test profile groups overlap.")

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing inputs require review.")

    for target in [y_train, y_test]:
        if not target.isin([0, 1]).all() or target.nunique() != 2:
            raise ValueError("Each partition must contain both target classes.")

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("binary", "passthrough", BINARY),
    ], remainder="drop")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(**config["model_parameters"])),
    ])

    output_dir = PROJECT_DIR / "pipeline_outputs"
    candidate_dir = output_dir / "candidate"
    tracking_dir = output_dir / "mlflow"
    artifact_dir = tracking_dir / "artifacts"

    candidate_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{tracking_dir / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    experiment_name = "Tourism_PreContact_Pipeline"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_dir.as_uri(),
        )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Fixed_Random_Forest_Training") as run:
        mlflow.set_tags({
            "project": "tourism_precontact",
            "purpose": "Reproduce selected model without tuning",
        })
        mlflow.log_params(config["model_parameters"])
        mlflow.log_params({
            "classification_threshold": threshold,
            "training_records": len(X_train),
            "test_records": len(X_test),
        })

        pipeline.fit(X_train, y_train)

        class_index = list(
            pipeline.named_steps["model"].classes_
        ).index(config["positive_class"])

        probabilities = pipeline.predict_proba(X_test)[:, class_index]
        predictions = (probabilities >= threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "average_precision": average_precision_score(y_test, probabilities),
        }

        mlflow.log_metrics({
            f"test_{name}": float(value)
            for name, value in metrics.items()
        })

        candidate_config = dict(config)
        # Preserve original run references as development provenance.
        candidate_config["pipeline_training_run_id"] = run.info.run_id

        joblib.dump(
            pipeline, candidate_dir / config["model_file"]
        )
        (candidate_dir / "model_config.json").write_text(
            json.dumps(candidate_config, indent=2), encoding="utf-8"
        )

        report = {
            "threshold": threshold,
            "metrics": {key: float(value) for key, value in metrics.items()},
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist(),
            "training_records": len(X_train),
            "test_records": len(X_test),
            "customers_flagged": int(predictions.sum()),
            "note": (
                "Reproduction of the fixed model on the existing test split; "
                "not a new independent evaluation."
            ),
        }

        (output_dir / "evaluation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        mlflow.log_artifacts(str(candidate_dir), artifact_path="candidate")
        mlflow.log_artifact(str(output_dir / "evaluation.json"))
        mlflow.log_artifact(str(config_path), artifact_path="source_configuration")

        mlflow.log_dict(
            {
                package: version(package)
                for package in [
                    "scikit-learn", "pandas", "numpy",
                    "scipy", "joblib", "mlflow",
                ]
            },
            "package_versions.json",
        )

        print("Pipeline training completed.")
        print("Run ID:", run.info.run_id)
        print(json.dumps(report, indent=2))
        print("Candidate model saved to:", candidate_dir)


if __name__ == "__main__":
    main()
