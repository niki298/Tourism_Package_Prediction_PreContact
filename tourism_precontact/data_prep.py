import argparse
import json
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_DIR = Path(__file__).resolve().parent

FEATURES = [
    "Age", "TypeofContact", "CityTier", "Occupation", "Gender",
    "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "processed",
    )
    args = parser.parse_args()

    data = pd.read_csv(PROJECT_DIR / "data" / "tourism.csv")

    # Correct the known spelling inconsistency.
    # Preserve Single/Unmarried and retain repeated profiles.
    data["Gender"] = data["Gender"].replace({"Fe Male": "Female"})

    X = data[FEATURES].copy()
    y = data["ProdTaken"].copy()

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Missing values require review.")

    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("The target must contain both classes 0 and 1.")

    # Group by predictors only, without using the outcome.
    groups = X.groupby(
        FEATURES,
        dropna=False,
        sort=False,
        observed=True,
    ).ngroup()

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if overlap:
        raise ValueError("Predictor profiles overlap between partitions.")

    if len(train_idx) + len(test_idx) != len(data):
        raise ValueError("The split did not retain all records.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_parts = []

    for partition, indices in [
        ("train", train_idx),
        ("test", test_idx),
    ]:
        X.iloc[indices].to_csv(
            output_dir / f"X_{partition}.csv", index=False
        )
        y.iloc[indices].rename("ProdTaken").to_csv(
            output_dir / f"y_{partition}.csv", index=False
        )
        groups.iloc[indices].rename("profile_group").to_csv(
            output_dir / f"groups_{partition}.csv", index=False
        )

        manifest_parts.append(pd.DataFrame({
            "source_row_index": X.index[indices].to_numpy(),
            "CustomerID": data["CustomerID"].iloc[indices].to_numpy(),
            "profile_group": groups.iloc[indices].to_numpy(),
            "partition": partition,
        }))

    pd.concat(manifest_parts, ignore_index=True).to_csv(
        output_dir / "split_manifest.csv", index=False
    )

    metadata = {
        "project": "tourism_precontact",
        "target": "ProdTaken",
        "feature_columns": FEATURES,
        "split_method": "First split from StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
        "group_definition": "Exact matches across the 14 selected predictors",
        "training_records": len(train_idx),
        "test_records": len(test_idx),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "availability_assumption": (
            "Selected predictors were recorded before the current "
            "campaign; the supplied dataset does not verify "
            "their collection timing."
        ),
    }

    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Data preparation: completed")
    print(f"Training records: {len(train_idx):,}")
    print(f"Test records: {len(test_idx):,}")
    print(f"Model inputs: {X.shape[1]}")
    print(f"Overlapping profiles: {len(overlap)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
