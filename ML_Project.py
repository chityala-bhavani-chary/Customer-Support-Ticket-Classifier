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
import pandas as pd                                       # ------------> to import the pandas library for data manipulation and analysis
import streamlit as st                                    # ------------> to import the Streamlit library for building the web application interface
import re                                                 # ------------> to import the regular expressions library for text cleaning and preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer                     # ------------> to import the TfidfVectorizer class from scikit-learn for converting text data into numerical features using the TF-IDF method
from sklearn.linear_model import LogisticRegression, LinearRegression           # ------------> to import the LogisticRegression and LinearRegression classes from scikit-learn for building classification and regression models
from sklearn.preprocessing import LabelEncoder                                  # ------------> to import the LabelEncoder class from scikit-learn for encoding categorical variables
from sklearn.metrics import accuracy_score, confusion_matrix                    # ------------> to import the accuracy_score and confusion_matrix functions from scikit-learn for evaluating the performance of classification models
from sklearn.model_selection import train_test_split                            # ------------> to import the train_test_split function from scikit-learn for splitting the dataset into training and testing sets

# -------------------------
# CLEAN TEXT
# -------------------------
from nltk.corpus import stopwords                       # ------------> to import the stopwords corpus from the Natural Language Toolkit (nltk) for removing common words that do not contribute much to the meaning of the text during text preprocessing
import nltk                                             # ------------> to import the Natural Language Toolkit (nltk) library for natural language processing tasks, including downloading the stopwords corpus

nltk.download('stopwords')                              # ------------> to download the stopwords corpus from nltk, which is necessary for removing common words during text preprocessing

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):                                   # ------------> to define a function called clean_text that takes a text input and performs cleaning and preprocessing tasks such as converting to lowercase, removing non-alphabetic characters, and filtering out stopwords
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    return " ".join(words)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data                                          # ------------> to use the Streamlit caching mechanism to cache the results of the load_data function, which loads and preprocesses the dataset, to improve performance by avoiding redundant computations when the function is called multiple times
def load_data():                                        # ------------> to define a function called load_data that loads the dataset from a CSV file, preprocesses the text data, and encodes categorical variables for use in machine learning models
    df = pd.read_csv("customer_support_tickets.csv")
    df.columns = df.columns.str.strip().str.lower()

    # Map columns
    df = df.rename(columns={
        'ticket_description': 'text',
        'ticket_subject': 'subject',
        'issue_category': 'category',
        'priority_level': 'priority',
        'resolution_time_hours': 'resolution_time'
    })

    df = df[['text', 'subject', 'category', 'priority', 'resolution_time']].dropna()

    df['cleaned'] = df['text'].apply(clean_text)

    # Encode
    le_p = LabelEncoder()                                 # ------------> to create an instance of the LabelEncoder class for encoding the 'priority' column in the dataset, which contains categorical values representing the priority levels of customer support tickets
    le_c = LabelEncoder()                                 # ------------> to create an instance of the LabelEncoder class for encoding the 'category' column in the dataset, which contains categorical values representing the issue categories of customer support tickets
    le_s = LabelEncoder()                                 # ------------> to create an instance of the LabelEncoder class for encoding the 'subject' column in the dataset, which contains categorical values representing the subjects of customer support tickets

    df['priority_enc'] = le_p.fit_transform(df['priority'])         # ------------> to encode the 'priority' column in the dataset using the fit_transform method of the LabelEncoder instance le_p, which converts the categorical values into numerical labels and stores the encoded values in a new column called 'priority_enc'
    df['category_enc'] = le_c.fit_transform(df['category'])         # ------------> to encode the 'category' column in the dataset using the fit_transform method of the LabelEncoder instance le_c, which converts the categorical values into numerical labels and stores the encoded values in a new column called 'category_enc'
    df['subject_enc'] = le_s.fit_transform(df['subject'])           # ------------> to encode the 'subject' column in the dataset using the fit_transform method of the LabelEncoder instance le_s, which converts the categorical values into numerical labels and stores the encoded values in a new column called 'subject_enc'

    return df, le_p, le_c, le_s

# -------------------------
# TRAIN MODELS
# -------------------------

@st.cache_resource
def train(df):

    tfidf = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2)
    )

    X = tfidf.fit_transform(
        df['cleaned']
    )

    # -------------------------
    # SUBJECT MODEL
    # -------------------------

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X,
        df['subject_enc'],
        test_size=0.2,
        random_state=42
    )

    model_s = LogisticRegression(
        max_iter=3000
    )

    model_s.fit(
        X_train_s,
        y_train_s
    )

    # -------------------------
    # PRIORITY MODEL
    # -------------------------

    model_p = LogisticRegression(
        max_iter=2000
    )

    model_p.fit(
        X,
        df['priority_enc']
    )

    # -------------------------
    # CATEGORY MODEL
    # -------------------------

    model_c = LogisticRegression(
        max_iter=2000
    )

    model_c.fit(
        X,
        df['category_enc']
    )

    # -------------------------
    # TIME MODEL
    # -------------------------

    model_t = LinearRegression()

    model_t.fit(
        X,
        df['resolution_time']
    )

    # -------------------------
    # ACCURACY
    # -------------------------

    accuracy = accuracy_score(
        df['priority_enc'],
        model_p.predict(X)
    )

    cm = confusion_matrix(
        df['priority_enc'],
        model_p.predict(X)
    )

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
# UI
# -------------------------
st.set_page_config(page_title="Ticket AI", layout="centered")
st.title("🎯 Smart Ticket Generator")

df, le_p, le_c, le_s = load_data()
tfidf, mp, mc, ms, mt, accuracy, cm = train(df)

text = st.text_area("✍️ Enter Customer Complaint")

if text.strip():

    x = tfidf.transform([clean_text(text)])

    # Predictions
    p = le_p.inverse_transform([mp.predict(x)[0]])[0]               #---> to predict the priority level of the customer complaint using the trained priority classification model (mp) and the TF-IDF features (x), and then inverse transform the predicted label back to its original categorical value using the LabelEncoder instance le_p, storing the result in the variable p
    c = le_c.inverse_transform([mc.predict(x)[0]])[0]               #---> to predict the issue category of the customer complaint using the trained category classification model (mc) and the TF-IDF features (x), and then inverse transform the predicted label back to its original categorical value using the LabelEncoder instance le_c, storing the result in the variable c
    s = le_s.inverse_transform([ms.predict(x)[0]])[0]               #---> to predict the ticket subject of the customer complaint using the trained subject classification model (ms) and the TF-IDF features (x), and then inverse transform the predicted label back to its original categorical value using the LabelEncoder instance le_s, storing the result in the variable s
    t = mt.predict(x)[0]

    # 🔥 UI OUTPUT
    st.subheader("Model Performance")                      
    st.write("Accuracy:", round(accuracy, 2))               #---> to display the accuracy of the model in the Streamlit app
    st.write("Confusion Matrix")                           
    st.write(cm)                                            #---> to display the confusion matrix in the Streamlit app
    st.markdown("## 🚨 Priority Level")
    st.error(p.upper())                                     #---> to display the predicted priority level of the customer complaint in the Streamlit app, formatted as an error message and converted to uppercase for emphasis

    st.markdown("### 🏷️ Issue Category")
    st.info(c)                                              #---> to display the predicted issue category of the customer complaint in the Streamlit app, formatted as an informational message

    st.markdown("### 📝 Ticket Subject")
    st.success(s)                                           #---> to display the predicted ticket subject of the customer complaint in the Streamlit app, formatted as a success message

    st.markdown("### ⏱️ Estimated Resolution Time")
    st.write(f"**{round(t,2)} hours**")                     #---> to display the predicted estimated resolution time for the customer complaint in the Streamlit app, formatted in bold and rounded to 2 decimal places, followed by the unit "hours"

else:
    st.info("Enter complaint to generate ticket")
