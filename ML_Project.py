# ML Project: Smart Ticket Classification & Resolution Time Prediction
# This project focuses on building a machine learning model to classify customer support tickets into priority levels, issue categories, and ticket subjects, as well as predicting the estimated resolution time. The model is trained on a dataset of customer support tickets and is designed to assist support teams in efficiently managing and prioritizing incoming tickets.
# The project includes the following steps:
# 1. Data Loading and Preprocessing: Load the dataset, clean the text data, and encode categorical variables.
# 2. Model Training: Train separate models for priority classification, category classification, subject classification, and resolution time prediction.
# 3. Model Evaluation: Evaluate the performance of the classification models using accuracy, confusion matrix, MAE, and R2 Score.
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
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- NLTK --------------------

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

# -------------------- CLEAN TEXT --------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)

    words = [
        w for w in text.split()
        if w not in STOPWORDS and len(w) > 2
    ]

    return " ".join(words)

# -------------------- LOAD DATA --------------------

@st.cache_data
def load_data():

    import zipfile
    with zipfile.ZipFile("customer_support_tickets_200k.zip") as z:
        with z.open("customer_support_tickets_200k.csv") as f:
            df = pd.read_csv(f)

    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        'issue_description'    : 'text',
        'product'              : 'subject',
        'category'             : 'category',
        'priority'             : 'priority',
        'resolution_time_hours': 'resolution_time'
    })

    df = df[
        ['text', 'subject', 'category', 'priority', 'resolution_time']
    ].dropna()

    # FIX 1: Normalize priority case (Low/low/LOW → Low)
    df['priority'] = df['priority'].str.strip().str.capitalize()

    # FIX 2: Keep only 4 valid priority levels (drops rare 'Critical' etc.)
    df = df[df['priority'].isin(['Low', 'Medium', 'High', 'Urgent'])]

    # FIX 3: Drop categories with fewer than 100 entries (removes noise)
    category_counts = df['category'].value_counts()
    valid_categories = category_counts[category_counts >= 100].index
    df = df[df['category'].isin(valid_categories)]

    df['cleaned'] = (df['subject'] + ' ' + df['text']).apply(clean_text)

    le_p = LabelEncoder()
    le_c = LabelEncoder()

    df['priority_enc'] = le_p.fit_transform(df['priority'])
    df['category_enc'] = le_c.fit_transform(df['category'])

    return df, le_p, le_c

# -------------------- TRAIN MODELS --------------------

@st.cache_resource
def train(df):

    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True,
        min_df=3
    )

    X = tfidf.fit_transform(df['cleaned'])

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X,
        df['priority_enc'],
        test_size=0.1,
        random_state=42
    )

    model_p = LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        C=5,
        solver='saga',
        n_jobs=-1
    )

    model_c = LogisticRegression(
        max_iter=500,
        solver='saga',
        n_jobs=-1
    )

    model_t = Ridge(alpha=1.0)

    model_p.fit(X_train_p, y_train_p)

    model_c.fit(X, df['category_enc'])

    model_t.fit(X, df['resolution_time'])

    pred_p = model_p.predict(X_test_p)

    accuracy = accuracy_score(y_test_p, pred_p)

    cm = confusion_matrix(y_test_p, pred_p)

    return tfidf, X, model_p, model_c, model_t, accuracy, cm

# -------------------- SUBJECT PREDICTION --------------------

def predict_subject(user_text, tfidf, X, df):

    sample_idx = df.sample(n=5000, random_state=42).index
    X_sample   = X[sample_idx]
    df_sample  = df.loc[sample_idx].reset_index(drop=True)

    user_vector = tfidf.transform(
        [clean_text(user_text)]
    )

    similarity = cosine_similarity(
        user_vector,
        X_sample
    )

    index = similarity.argmax()

    return df_sample.iloc[index]['subject']

# -------------------- UI --------------------

st.set_page_config(
    page_title="Ticket AI",
    layout="centered"
)

st.title("🎯 Smart Ticket Generator")

df, le_p, le_c = load_data()

tfidf, X, mp, mc, mt, accuracy, cm = train(df)

text = st.text_area(
    "✍️ Enter Customer Complaint"
)

# -------------------- PREDICTIONS --------------------

if text.strip():

    x = tfidf.transform(
        [clean_text(text)]
    )

    p = le_p.inverse_transform(
        [mp.predict(x)[0]]
    )[0]

    c = le_c.inverse_transform(
        [mc.predict(x)[0]]
    )[0]

    s = predict_subject(text, tfidf, X, df)

    t = mt.predict(x)[0]

    # -------------------- OUTPUT --------------------

    st.subheader("📊 Model Performance")

    st.write(
        "Accuracy:",
        round(accuracy, 2)
    )

    st.write("Confusion Matrix")

    st.write(cm)

    st.markdown("## 🚨 Priority Level")

    st.error(p.upper())

    st.markdown("### 🏷️ Issue Category")

    st.info(c)

    st.markdown("### 📝 Product")

    st.success(s)

    st.markdown("### ⏱️ Estimated Resolution Time")

    st.write(
        f"**{round(float(t), 2)} hours**"
    )

else:

    st.info(
        "Enter complaint to generate ticket"
    )
