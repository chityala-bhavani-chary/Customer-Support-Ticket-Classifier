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
import pandas as pd
import streamlit as st
import re
import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# -------------------------
# NLTK Setup
# -------------------------

try:
    nltk.data.find('corpora/stopwords')

except:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

# -------------------------
# Clean Text
# -------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = [
        w for w in text.split()
        if w not in STOPWORDS and len(w) > 2
    ]

    return " ".join(words)

# -------------------------
# Load Data
# -------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("customer_support_tickets.csv")

    df.columns = df.columns.str.strip().str.lower()

    # Rename Columns

    df = df.rename(columns={
        'ticket_description': 'text',
        'ticket_subject': 'subject',
        'issue_category': 'category',
        'priority_level': 'priority',
        'resolution_time_hours': 'resolution_time'
    })

    # Required Columns

    df = df[
        [
            'text',
            'subject',
            'category',
            'priority',
            'resolution_time'
        ]
    ].dropna()

    # Keep only top 15 frequent subjects

    top_subjects = (
        df['subject']
        .value_counts()
        .nlargest(15)
        .index
    )

    df = df[
        df['subject'].isin(top_subjects)
    ]

    # Clean Text

    df['cleaned'] = df['text'].apply(clean_text)

    # Label Encoding

    le_p = LabelEncoder()

    le_c = LabelEncoder()

    le_s = LabelEncoder()

    df['priority_enc'] = le_p.fit_transform(
        df['priority']
    )

    df['category_enc'] = le_c.fit_transform(
        df['category']
    )

    df['subject_enc'] = le_s.fit_transform(
        df['subject']
    )

    return df, le_p, le_c, le_s

# -------------------------
# Train Models
# -------------------------

@st.cache_resource
def train(df):

    tfidf = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2)
    )

    X = tfidf.fit_transform(df['cleaned'])

    # -------------------------
    # Priority Model
    # -------------------------

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X,
        df['priority_enc'],
        test_size=0.3,
        random_state=42
    )

    model_p = LogisticRegression(max_iter=2000)

    model_p.fit(X_train_p, y_train_p)

    pred_p = model_p.predict(X_test_p)

    accuracy = accuracy_score(
        y_test_p,
        pred_p
    )

    cm = confusion_matrix(
        y_test_p,
        pred_p
    )

    # -------------------------
    # Category Model
    # -------------------------

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X,
        df['category_enc'],
        test_size=0.3,
        random_state=42
    )

    model_c = LogisticRegression(max_iter=2000)

    model_c.fit(X_train_c, y_train_c)

    # -------------------------
    # Subject Model
    # -------------------------

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X,
        df['subject_enc'],
        test_size=0.3,
        random_state=42
    )

    model_s = LogisticRegression(max_iter=2000)

    model_s.fit(X_train_s, y_train_s)

    # -------------------------
    # Resolution Time Model
    # -------------------------

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X,
        df['resolution_time'],
        test_size=0.3,
        random_state=42
    )

    model_t = LinearRegression()

    model_t.fit(X_train_t, y_train_t)

    return (
        tfidf,
        model_p,
        model_c,
        model_s,
        model_t,
        accuracy,
        cm
    )

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(
    page_title="Ticket AI",
    layout="centered"
)

st.title("🎯 Smart Ticket Classifier")

# Load Data

df, le_p, le_c, le_s = load_data()

# Train Models

(
    tfidf,
    mp,
    mc,
    ms,
    mt,
    accuracy,
    cm
) = train(df)

# User Input

text = st.text_area(
    "✍️ Enter Customer Complaint"
)

# -------------------------
# Prediction
# -------------------------

if text.strip():

    x = tfidf.transform(
        [clean_text(text)]
    )

    # Predictions

    p = le_p.inverse_transform(
        [mp.predict(x)[0]]
    )[0]

    c = le_c.inverse_transform(
        [mc.predict(x)[0]]
    )[0]

    s = le_s.inverse_transform(
        [ms.predict(x)[0]]
    )[0]

    t = mt.predict(x)[0]

    # -------------------------
    # UI Output
    # -------------------------

    st.subheader("📊 Model Performance")

    st.write(
        "Priority Accuracy:",
        round(accuracy, 2)
    )

    st.write("Confusion Matrix")

    st.write(cm)

    st.markdown("## 🚨 Priority Level")

    st.error(p.upper())

    st.markdown("### 🏷️ Issue Category")

    st.info(c)

    st.markdown("### 📝 Ticket Subject")

    st.success(s)

    st.markdown("### ⏱️ Estimated Resolution Time")

    st.write(
        f"**{round(t, 2)} hours**"
    )

else:

    st.info(
        "Enter complaint to generate ticket"
    )
