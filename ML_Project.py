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


# -------------------------
# IMPORTS
# -------------------------
import pandas as pd                                       # ------------> to import the pandas library for data manipulation and analysis
import streamlit as st                                    # ------------> to import the Streamlit library for building the web application interface
import re                                                 # ------------> to import the regular expressions library for text cleaning and preprocessing
import nltk                                               # ------------> to import the Natural Language Toolkit (nltk) library for natural language processing tasks

from nltk.corpus import stopwords                         # ------------> to import the stopwords corpus from nltk for removing common words
from nltk.stem import PorterStemmer                       # ------------> to import the PorterStemmer class from nltk for word stemming
from sklearn.feature_extraction.text import TfidfVectorizer                     # ------------> to import the TfidfVectorizer class from scikit-learn for converting text data into numerical features using the TF-IDF method
from sklearn.linear_model import LogisticRegression, LinearRegression           # ------------> to import the LogisticRegression and LinearRegression classes from scikit-learn for building classification and regression models
from sklearn.preprocessing import LabelEncoder                                  # ------------> to import the LabelEncoder class from scikit-learn for encoding categorical variables
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score # ------------> to import evaluation metrics for both classification and regression models
from sklearn.model_selection import train_test_split                            # ------------> to import the train_test_split function from scikit-learn for splitting the dataset into training and testing sets

# -------------------------
# CLEAN TEXT
# -------------------------
@st.cache_resource
def load_nlp_utils():                                   # ------------> to download necessary NLTK data and initialize the stemmer and stopwords set once
    nltk.download('stopwords')
    return set(stopwords.words('english')), PorterStemmer()

STOPWORDS, STEMMER = load_nlp_utils()

def clean_text(text):                                   # ------------> to define a function called clean_text that performs cleaning, tokenization, stopword removal, and stemming to prepare text for vectorization
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    words = [STEMMER.stem(w) for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():

    df = pd.read_csv(
        "customer_support_tickets.csv"
    )

    # Clean column names

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # 🔥 Show actual dataset columns

    st.write("Dataset Columns:", df.columns.tolist())

    # Rename only if columns exist

    rename_map = {}

    if 'ticket_description' in df.columns:
        rename_map['ticket_description'] = 'text'

    if 'ticket_subject' in df.columns:
        rename_map['ticket_subject'] = 'subject'

    if 'issue_category' in df.columns:
        rename_map['issue_category'] = 'category'

    if 'priority_level' in df.columns:
        rename_map['priority_level'] = 'priority'

    if 'resolution_time_hours' in df.columns:
        rename_map['resolution_time_hours'] = 'resolution_time'

    df = df.rename(columns=rename_map)

    # ✅ Check required columns

    required = [
        'text',
        'subject',
        'category',
        'priority',
        'resolution_time'
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        st.error(
            f"Missing Columns: {missing}"
        )

        st.stop()

    # Keep required columns

    df = df[
        required
    ].dropna()

    # 🔥 Keep only Top 10 subjects

    top_subjects = (
        df['subject']
        .value_counts()
        .nlargest(10)
        .index
    )

    df = df[
        df['subject'].isin(top_subjects)
    ]

    # Clean text

    df['cleaned'] = df['text'].apply(
        clean_text
    )

    # Encoding

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
# TRAIN MODELS
# -------------------------
@st.cache_resource
def train(df):          # ------------> to define a function called train that vectorizes text and trains Logistic and Linear regression models, utilizing 'balanced' class weights to improve accuracy
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))       # ------------> to create a TF-IDF vectorizer considering unigrams and bigrams to capture more context from the support tickets
    X = tfidf.fit_transform(df['cleaned'])

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X,
    df['subject_enc'],
    test_size=0.2,
    random_state=42
    )
    
    model_s = LogisticRegression(
        class_weight='balanced',
        max_iter=2000
    )
    
    model_s.fit(
        X_train_s,
        y_train_s
    )            # ------------> to split the data for validation, using 20% for testing

    # Fixed: Removed multi_class argument to resolve TypeError in scikit-learn 1.8.0
    model_p = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000).fit(X_train, y_train_p) 
    model_c = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X, df['category_enc'])
    model_t = LinearRegression().fit(X, df['resolution_time'])                      

    # Evaluation Metrics
    accuracy = accuracy_score(y_test_p, model_p.predict(X_test))               # ------------> to calculate accuracy score for priority classification
    cm = confusion_matrix(y_test_p, model_p.predict(X_test))                   # ------------> to compute the confusion matrix for the priority classification model
    mae = mean_absolute_error(df['resolution_time'], model_t.predict(X))        # ------------> to calculate the Mean Absolute Error for the resolution time regression model
    r2 = r2_score(df['resolution_time'], model_t.predict(X))                    # ------------> to calculate the R2 Score for the regression model
    
    return tfidf, model_p, model_c, model_s, model_t, accuracy, cm, mae, r2

# -------------------------
# EXECUTION & CACHING
# -------------------------
@st.cache_resource
def get_everything():                                   # ------------> to wrap data loading and training in a single cached function to fix rerunning delays and potential NameErrors
    df, le_p, le_c, le_s = load_data()
    tfidf, mp, mc, ms, mt, accuracy, cm, mae, r2 = train(df)
    return df, le_p, le_c, le_s, tfidf, mp, mc, ms, mt, accuracy, cm, mae, r2

# --- Unpack all resources ---
df, le_p, le_c, le_s, tfidf, mp, mc, ms, mt, accuracy, cm, mae, r2 = get_everything()

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Ticket AI", layout="centered")
st.title("🎯 Smart Ticket Generator")

text = st.text_area("✍️ Enter Customer Complaint")

if text.strip():

    x = tfidf.transform([clean_text(text)])

    # Predictions
    p = le_p.inverse_transform([mp.predict(x)[0]])[0]               #---> to predict the priority level and inverse transform it to original text
    c = le_c.inverse_transform([mc.predict(x)[0]])[0]               #---> to predict the issue category and inverse transform it to original text
    s = le_s.inverse_transform([ms.predict(x)[0]])[0]               #---> to predict the ticket subject and inverse transform it to original text
    t = mt.predict(x)[0]

    # 🔥 UI OUTPUT
    st.subheader("Model Performance")                      
    st.write("Accuracy:", round(accuracy, 2))               #---> to display the accuracy of the model in the Streamlit app
    st.write("Regression R2 Score:", round(r2, 2))          #---> to display the R-squared value for regression
    st.write("Confusion Matrix")                           
    st.write(cm)                                            #---> to display the confusion matrix in the Streamlit app
    
    st.markdown("## 🚨 Priority Level")
    st.error(p.upper())                                     #---> to display the predicted priority level formatted as an error message

    st.markdown("### 🏷️ Issue Category")
    st.info(c)                                              #---> to display the predicted issue category formatted as an info message

    st.markdown("### 📝 Ticket Subject")
    st.success(s)                                           #---> to display the predicted ticket subject formatted as a success message

    st.markdown("### ⏱️ Estimated Resolution Time")
    st.write(f"**{round(float(t),2)} hours** (MAE: {round(mae, 2)})") #---> to display predicted time and error margin in the Streamlit app

else:
    st.info("Enter complaint to generate ticket")
