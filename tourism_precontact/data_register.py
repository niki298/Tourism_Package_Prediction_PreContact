import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = PROJECT_DIR / "data" / "tourism.csv"
REPORT_PATH = PROJECT_DIR / "reports" / "data_validation.json"

EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def main():
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_PATH}")

    data = pd.read_csv(SOURCE_PATH)

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    unexpected_columns = sorted(set(data.columns) - set(EXPECTED_COLUMNS))

    if missing_columns or unexpected_columns:
        raise ValueError(
            f"Schema mismatch. Missing: {missing_columns}; "
            f"unexpected: {unexpected_columns}"
        )

    if data.empty:
        raise ValueError("The dataset contains no records.")

    if data.isna().any().any():
        missing_counts = data.isna().sum()
        raise ValueError(
            "Missing values require review before using the current "
            f"pipeline: {missing_counts[missing_counts > 0].to_dict()}"
        )

    if not data["ProdTaken"].isin([0, 1]).all():
        raise ValueError("ProdTaken must contain only 0 and 1.")

    if data["ProdTaken"].nunique() != 2:
        raise ValueError("Both target classes must be present.")

    if data["CustomerID"].duplicated().any():
        raise ValueError("Repeated CustomerID values require review.")

    # SHA-256 identifies the exact source file used by this run.
    fingerprint = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()

    report = {
        "validation_status": "passed",
        "source_file": str(SOURCE_PATH.relative_to(PROJECT_DIR)),
        "sha256": fingerprint,
        "records": int(len(data)),
        "columns": int(data.shape[1]),
        "missing_values": int(data.isna().sum().sum()),
        "unique_customer_ids": int(data["CustomerID"].nunique()),
        "non_purchasers": int((data["ProdTaken"] == 0).sum()),
        "purchasers": int((data["ProdTaken"] == 1).sum()),
        "note": (
            "Validation confirms the checked structural requirements. "
            "It does not verify pre-contact feature availability."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Dataset validation: PASSED")
    print(f"Records: {len(data):,}")
    print(f"Columns: {data.shape[1]}")
    print(f"Non-purchasers: {report['non_purchasers']:,}")
    print(f"Purchasers: {report['purchasers']:,}")
    print(f"Source SHA-256: {fingerprint}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
