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


# ML Project: Smart Ticket Classification & Resolution Time Prediction

# -------------------------
# IMPORTS
# -------------------------

import pandas as pd
import streamlit as st
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# CLEAN TEXT
# -------------------------

from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find('corpora/stopwords')

except:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()

    words = [
        w for w in words
        if w not in STOPWORDS and len(w) > 2
    ]

    return " ".join(words)

# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "customer_support_tickets.csv"
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Map columns

    df = df.rename(columns={

        'ticket_description': 'text',

        'ticket_subject': 'subject',

        'issue_category': 'category',

        'priority_level': 'priority',

        'resolution_time_hours': 'resolution_time'
    })

    df = df[
        [
            'text',
            'subject',
            'category',
            'priority',
            'resolution_time'
        ]
    ].dropna()

    # Clean text

    df['cleaned'] = df['text'].apply(
        clean_text
    )

    # Encode

    le_p = LabelEncoder()

    le_c = LabelEncoder()

    df['priority_enc'] = le_p.fit_transform(
        df['priority']
    )

    df['category_enc'] = le_c.fit_transform(
        df['category']
    )

    return df, le_p, le_c

# -------------------------
# TRAIN MODELS
# -------------------------

@st.cache_resource
def train(df):

    tfidf = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1,2)
    )

    X = tfidf.fit_transform(
        df['cleaned']
    )

    # -------------------------
    # Priority Model
    # -------------------------

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X,
        df['priority_enc'],
        test_size=0.3,
        random_state=42
    )

    model_p = LogisticRegression(
        max_iter=3000
    )

    model_p.fit(
        X_train_p,
        y_train_p
    )

    pred_p = model_p.predict(
        X_test_p
    )

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

    model_c = LogisticRegression(
        max_iter=3000
    )

    model_c.fit(
        X,
        df['category_enc']
    )

    # -------------------------
    # Resolution Time Model
    # -------------------------

    model_t = LinearRegression()

    model_t.fit(
        X,
        df['resolution_time']
    )

    return (
        tfidf,
        X,
        model_p,
        model_c,
        model_t,
        accuracy,
        cm
    )

# -------------------------
# RELATED SUBJECTS
# -------------------------

def get_related_subjects(
    user_text,
    df,
    tfidf,
    X,
    top_n=5
):

    user_vector = tfidf.transform(
        [clean_text(user_text)]
    )

    similarity = cosine_similarity(
        user_vector,
        X
    )

    top_indices = (
        similarity[0]
        .argsort()[-top_n:][::-1]
    )

    subjects = (
        df.iloc[top_indices]['subject']
        .unique()
    )

    return subjects

# -------------------------
# UI
# -------------------------

st.set_page_config(
    page_title="Ticket AI",
    layout="centered"
)

st.title("🎯 Smart Ticket Generator")

df, le_p, le_c = load_data()

(
    tfidf,
    X,
    mp,
    mc,
    mt,
    accuracy,
    cm
) = train(df)

text = st.text_area(
    "✍️ Enter Customer Complaint"
)

# -------------------------
# PREDICTIONS
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

    t = mt.predict(x)[0]

    # Related Subjects

    related_subjects = get_related_subjects(
        text,
        df,
        tfidf,
        X
    )

    # -------------------------
    # UI OUTPUT
    # -------------------------

    st.subheader("Model Performance")

    st.write(
        "Accuracy:",
        round(accuracy, 2)
    )

    st.write("Confusion Matrix")

    st.write(cm)

    # Priority

    st.markdown(
        "## 🚨 Priority Level"
    )

    st.error(
        p.upper()
    )

    # Category

    st.markdown(
        "### 🏷️ Issue Category"
    )

    st.info(c)

    # Related Subjects

    st.markdown(
        "### 📝 Related Ticket Subjects"
    )

    for subject in related_subjects:

        st.success(subject)

    # Resolution Time

    st.markdown(
        "### ⏱️ Estimated Resolution Time"
    )

    st.write(
        f"**{round(t,2)} hours**"
    )

else:

    st.info(
        "Enter complaint to generate ticket"
    )
