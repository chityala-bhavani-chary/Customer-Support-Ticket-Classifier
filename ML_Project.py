# ML Project: Smart Ticket Classification & Resolution Time Prediction
# This project focuses on building a machine learning model to classify customer support tickets into priority levels, issue categories, and ticket subjects, as well as predicting the estimated resolution time. The model is trained on a dataset of customer support tickets and is designed to assist support teams in efficiently managing and prioritizing incoming tickets.
# The project includes the following steps:
# 1. Data Loading and Preprocessing: Load the dataset, clean the text data,
#    and encode categorical variables.
# 2. Model Training: Train separate models for priority classification, category classification, subject classification, and resolution time prediction.
# 3. Model Evaluation: Evaluate the performance of the classification models using accuracy and confusion matrix.
# 4. Streamlit UI: Create an interactive user interface using Streamlit to allow users to input customer complaints and receive predictions for priority level, issue category, ticket subject, and estimated resolution time.
# The project utilizes libraries such as pandas for data manipulation, scikit-learn for machine learning, and Streamlit for building the web application interface.
# Note: Ensure that the dataset "customer_support_tickets.csv" is available in the same directory as this script for it to run successfully.
# Import necessary libraries


# -------------------------
# IMPORTS
# -------------------------
# -------------------------
# IMPORTS
# -------------------------

import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# -------------------------
# NLTK SETUP
# -------------------------

try:
    nltk.data.find("corpora/stopwords")

except:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

# -------------------------
# TEXT CLEANING
# -------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = [
        word for word in text.split()
        if word not in STOPWORDS and len(word) > 2
    ]

    return " ".join(words)

# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data():

    try:

        df = pd.read_csv(
            "customer_support_tickets.csv"
        )

    except Exception as e:

        st.error(
            f"Dataset Error: {e}"
        )

        st.stop()

    # Clean column names

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Rename columns

    rename_map = {

        "ticket_description": "text",

        "ticket_subject": "subject",

        "issue_category": "category",

        "priority_level": "priority",

        "resolution_time_hours": "resolution_time"
    }

    df = df.rename(columns=rename_map)

    # Required columns

    required_columns = [

        "text",

        "subject",

        "category",

        "priority",

        "resolution_time"
    ]

    # Check missing columns

    missing = [

        col for col in required_columns

        if col not in df.columns
    ]

    if missing:

        st.error(
            f"Missing Columns: {missing}"
        )

        st.stop()

    # Keep needed columns

    df = df[
        required_columns
    ].dropna()

    # Clean text

    df["cleaned"] = df["text"].apply(
        clean_text
    )

    # Label Encoding

    le_priority = LabelEncoder()

    le_category = LabelEncoder()

    le_subject = LabelEncoder()

    df["priority_enc"] = le_priority.fit_transform(
        df["priority"]
    )

    df["category_enc"] = le_category.fit_transform(
        df["category"]
    )

    df["subject_enc"] = le_subject.fit_transform(
        df["subject"]
    )

    return (
        df,
        le_priority,
        le_category,
        le_subject
    )

# -------------------------
# TRAIN MODELS
# -------------------------

@st.cache_resource
def train_models(df):

    tfidf = TfidfVectorizer(
        max_features=1000
    )

    X = tfidf.fit_transform(
        df["cleaned"]
    )

    # -------------------------
    # PRIORITY MODEL
    # -------------------------

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X,
        df["priority_enc"],
        test_size=0.3,
        random_state=42
    )

    priority_model = LogisticRegression(
        max_iter=2000
    )

    priority_model.fit(
        X_train_p,
        y_train_p
    )

    priority_pred = priority_model.predict(
        X_test_p
    )

    accuracy = accuracy_score(
        y_test_p,
        priority_pred
    )

    cm = confusion_matrix(
        y_test_p,
        priority_pred
    )

    # -------------------------
    # CATEGORY MODEL
    # -------------------------

    category_model = LogisticRegression(
        max_iter=2000
    )

    category_model.fit(
        X,
        df["category_enc"]
    )

    # -------------------------
    # SUBJECT MODEL
    # -------------------------

    subject_model = LogisticRegression(
        max_iter=2000
    )

    subject_model.fit(
        X,
        df["subject_enc"]
    )

    # -------------------------
    # RESOLUTION TIME MODEL
    # -------------------------

    time_model = LinearRegression()

    time_model.fit(
        X,
        df["resolution_time"]
    )

    return (
        tfidf,
        priority_model,
        category_model,
        subject_model,
        time_model,
        accuracy,
        cm
    )

# -------------------------
# LOAD EVERYTHING
# -------------------------

(
    df,
    le_priority,
    le_category,
    le_subject
) = load_data()

(
    tfidf,
    priority_model,
    category_model,
    subject_model,
    time_model,
    accuracy,
    cm
) = train_models(df)

# -------------------------
# STREAMLIT UI
# -------------------------

st.set_page_config(
    page_title="Smart Ticket Classifier",
    layout="centered"
)

st.title("🎯 Smart Ticket Classifier")

st.write(
    "Predict ticket priority, category, subject, and estimated resolution time."
)

user_text = st.text_area(
    "✍️ Enter Customer Complaint"
)

# -------------------------
# PREDICTION
# -------------------------

if st.button("Predict"):

    if not user_text.strip():

        st.warning(
            "Please enter complaint text."
        )

    else:

        cleaned = clean_text(
            user_text
        )

        x = tfidf.transform(
            [cleaned]
        )

        # Predictions

        priority = le_priority.inverse_transform(
            [priority_model.predict(x)[0]]
        )[0]

        category = le_category.inverse_transform(
            [category_model.predict(x)[0]]
        )[0]

        subject = le_subject.inverse_transform(
            [subject_model.predict(x)[0]]
        )[0]

        resolution_time = time_model.predict(
            x
        )[0]

        # -------------------------
        # RESULTS
        # -------------------------

        st.subheader(
            "📊 Prediction Results"
        )

        st.error(
            f"🚨 Priority: {priority}"
        )

        st.info(
            f"🏷️ Category: {category}"
        )

        st.success(
            f"📝 Subject: {subject}"
        )

        st.write(
            f"⏱️ Estimated Resolution Time: "
            f"{round(resolution_time, 2)} hours"
        )

        st.subheader(
            "📈 Model Accuracy"
        )

        st.write(
            f"Priority Model Accuracy: "
            f"{round(accuracy, 2)}"
        )

        st.subheader(
            "📌 Confusion Matrix"
        )

        st.write(cm)
