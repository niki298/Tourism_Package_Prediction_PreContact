import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Visit With Us — Pre-Contact Targeting",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_assets():
    with (BASE_DIR / "model_config.json").open() as file:
        config = json.load(file)

    pipeline = joblib.load(BASE_DIR / config["model_file"])
    return pipeline, config


pipeline, config = load_assets()

st.title("Visit With Us")
st.subheader("Pre-Contact Tourism Package Targeting")

st.write(
    "Use existing customer records to prioritise customers "
    "for a tourism package campaign."
)

st.info(
    "Enter only information recorded before the first contact "
    "for the current campaign. Travel details must describe "
    "historical records, and contact type must represent a "
    "previously recorded acquisition source."
)

with st.form("customer_details"):
    st.subheader("Customer profile")

    age = st.number_input("Age", min_value=18, value=35, step=1)
    city = st.selectbox("City tier", [1, 2, 3])
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    )
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital = st.selectbox(
        "Marital status",
        ["Married", "Single", "Unmarried", "Divorced"],
    )
    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    )
    income = st.number_input(
        "Monthly income (same currency and units as the training data)",
        min_value=0.0,
        value=23000.0,
        step=500.0,
    )

    st.subheader("Historical travel information")

    people = st.number_input(
        "Historical travel-party size, including children",
        min_value=1,
        value=3,
        step=1,
    )
    children = st.number_input(
        "Historical number of children below age 5 travelling",
        min_value=0,
        value=1,
        step=1,
    )
    property_star = st.selectbox(
        "Historical preferred property star rating",
        [3, 4, 5],
    )
    trips = st.number_input(
        "Recorded number of trips per year",
        min_value=0,
        value=3,
        step=1,
    )
    passport = st.selectbox("Has a passport?", ["No", "Yes"])
    car = st.selectbox("Owns a car?", ["No", "Yes"])

    st.subheader("Historical acquisition source")

    contact = st.selectbox(
        "Previously recorded contact type",
        ["Self Enquiry", "Company Invited"],
    )

    availability_confirmed = st.checkbox(
        "I confirm that these details were available before "
        "the current campaign."
    )

    submitted = st.form_submit_button("Assess targeting priority")

if submitted:
    errors = []

    if not availability_confirmed:
        errors.append(
            "Confirm that all inputs were available before "
            "the current campaign."
        )

    if children >= people:
        errors.append(
            "The travel party must include at least one person "
            "other than the children below age 5."
        )

    if errors:
        for message in errors:
            st.error(message)
    else:
        inputs = pd.DataFrame([{
            "Age": float(age),
            "TypeofContact": contact,
            "CityTier": int(city),
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": int(people),
            "PreferredPropertyStar": float(property_star),
            "MaritalStatus": marital,
            "NumberOfTrips": float(trips),
            "Passport": int(passport == "Yes"),
            "OwnCar": int(car == "Yes"),
            "NumberOfChildrenVisiting": float(children),
            "Designation": designation,
            "MonthlyIncome": float(income),
        }])

        inputs = inputs[config["feature_columns"]]

        # Warn about numerical values outside the training ranges.
        outside_range = (
            age > 61
            or income < 1000
            or income > 98678
            or trips < 1
            or trips > 22
            or people > 5
            or children > 3
        )

        if outside_range:
            st.warning(
                "Some values fall outside the ranges observed in "
                "the training data. Predictions may be less reliable."
            )

        class_index = list(
            pipeline.named_steps["model"].classes_
        ).index(config["positive_class"])

        score = float(
            pipeline.predict_proba(inputs)[0, class_index]
        )
        threshold = float(config["classification_threshold"])
        flagged = score >= threshold

        st.metric("Model purchase score", f"{score:.3f}")

        if flagged:
            st.success("Flagged for targeting")
        else:
            st.info("Not flagged at the selected threshold")

        st.caption(
            f"Targeting threshold: {threshold:.2f}. "
            "The score is a model estimate, not a guaranteed outcome "
            "or a calibrated purchase probability."
        )

st.caption(
    "Learning prototype for existing customers. "
    "Use human review before campaign decisions."
)
