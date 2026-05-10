# ML Project: Smart Ticket Classification & Resolution Time Prediction
# This project focuses on building a machine learning model to classify customer support tickets into priority levels, issue categories, and ticket subjects, as well as predicting the estimated resolution time. The model is trained on a dataset of customer support tickets and is designed to assist support teams in efficiently managing and prioritizing incoming tickets.
# The project includes the following steps:
# 1. Data Loading and Preprocessing: Load the dataset, clean the text data,
#    and encode categorical variables.
# 2. Model Training: Train separate models for priority classification, category classification, subject classification, and resolution time prediction.
# 3. Model Evaluation: Evaluate the performance of the classification models using accuracy and confusion matrix.
# 4. Streamlit UI: Create an interactive user interface using Streamlit to allow users to input customer complaints and receive predictions for priority level, issue category, ticket subject, and estimated resolution time.
# The project utilizes libraries such as pandas for data manipulation, scikit-learn for machine learning, and Streamlit for building the web application interface.
# Note: Ensure that the dataset "customer_support_tickets_200k.zip" is available in the same directory as this script for it to run successfully.
# Import necessary libraries


# -------------------------
# IMPORTS
# -------------------------
import pandas as pd                                       # ------------> to import the pandas library for data manipulation and analysis
import streamlit as st                                    # ------------> to import the Streamlit library for building the web application interface
import re                                                 # ------------> to import the regular expressions library for text cleaning and preprocessing
import numpy as np                                        # ------------> to import numpy for numerical operations

from sklearn.feature_extraction.text import TfidfVectorizer                     # ------------> to import the TfidfVectorizer class from scikit-learn for converting text data into numerical features using the TF-IDF method
from sklearn.linear_model import LogisticRegression, LinearRegression           # ------------> to import the LogisticRegression and LinearRegression classes from scikit-learn for building classification and regression models
from sklearn.preprocessing import LabelEncoder                                  # ------------> to import the LabelEncoder class from scikit-learn for encoding categorical variables
from sklearn.metrics import accuracy_score, confusion_matrix                    # ------------> to import the accuracy_score and confusion_matrix functions from scikit-learn for evaluating the performance of classification models
from sklearn.model_selection import train_test_split                            # ------------> to import the train_test_split function from scikit-learn for splitting the dataset into training and testing sets
from sklearn.metrics.pairwise import cosine_similarity                          # ------------> to import cosine_similarity for finding the most similar ticket subject

# -------------------------
# CLEAN TEXT
# -------------------------
from nltk.corpus import stopwords                       # ------------> to import the stopwords corpus from the Natural Language Toolkit (nltk) for removing common words that do not contribute much to the meaning of the text during text preprocessing
import nltk                                             # ------------> to import the Natural Language Toolkit (nltk) library for natural language processing tasks, including downloading the stopwords corpus

nltk.download('stopwords', quiet=True)                  # ------------> to download the stopwords corpus from nltk, which is necessary for removing common words during text preprocessing

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):                                   # ------------> to define a function called clean_text that takes a text input and performs cleaning and preprocessing tasks such as converting to lowercase, removing non-alphabetic characters, and filtering out stopwords
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    return " ".join(words)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data                                          # ------------> to use the Streamlit caching mechanism to cache the results of the load_data function, which loads and preprocesses the dataset, to improve performance by avoiding redundant computations when the function is called multiple times
def load_data():                                        # ------------> to define a function called load_data that loads the dataset from a CSV file, preprocesses the text data, and encodes categorical variables for use in machine learning models

    import zipfile
    with zipfile.ZipFile("customer_support_tickets_200k.zip") as z:         # ------------> to open the zip file containing the dataset
        with z.open("customer_support_tickets_200k.csv") as f:              # ------------> to open the CSV file inside the zip, avoiding Mac __MACOSX folder issues
            df = pd.read_csv(f)

    df.columns = df.columns.str.strip().str.lower()

    # Map columns to match original code naming convention
    df = df.rename(columns={
        'issue_description'    : 'text',
        'product'              : 'subject',
        'category'             : 'category',
        'priority'             : 'priority',
        'resolution_time_hours': 'resolution_time'
    })

    df = df[['text', 'subject', 'category', 'priority', 'resolution_time']].dropna()

    # Normalize priority case so Low/low/LOW are treated as one class
    df['priority'] = df['priority'].str.strip().str.capitalize()

    # Keep only 4 valid priority levels — drops rare/invalid entries
    df = df[df['priority'].isin(['Low', 'Medium', 'High', 'Urgent'])]

    # Drop categories with fewer than 100 entries to remove noise
    category_counts  = df['category'].value_counts()
    valid_categories = category_counts[category_counts >= 100].index
    df               = df[df['category'].isin(valid_categories)]

    # Reset index so it always aligns with X matrix row positions (fixes IndexError)
    df = df.reset_index(drop=True)

    # Combine subject + text for richer features (doubles signal strength)
    df['cleaned'] = (df['subject'] + ' ' + df['text']).apply(clean_text)

    le_p = LabelEncoder()                                 # ------------> to create an instance of the LabelEncoder class for encoding the 'priority' column
    le_c = LabelEncoder()                                 # ------------> to create an instance of the LabelEncoder class for encoding the 'category' column

    df['priority_enc'] = le_p.fit_transform(df['priority'])         # ------------> to encode the 'priority' column into numerical labels
    df['category_enc'] = le_c.fit_transform(df['category'])         # ------------> to encode the 'category' column into numerical labels

    return df, le_p, le_c

# -------------------------
# TRAIN MODELS
# -------------------------
@st.cache_resource
def train(df):          # ------------> to define a function called train that takes the preprocessed dataset as input, trains machine learning models for priority classification, category classification, subject prediction, and resolution time prediction, and returns the trained models along with evaluation metrics

    tfidf = TfidfVectorizer(
        max_features=30000,     # ------------> increased from 5000 to capture more discriminative terms
        ngram_range=(1, 2),     # ------------> bigrams capture phrases like "payment failed", "login issue"
        stop_words='english',
        sublinear_tf=True,      # ------------> log-scale TF dampens very frequent generic words
        min_df=2                # ------------> ignore terms appearing in only 1 document (noise)
    )

    X = tfidf.fit_transform(df['cleaned'])

    X_train, X_test, y_train_p, y_test_p = train_test_split(
        X, df['priority_enc'], test_size=0.1, random_state=42
    )   # ------------> to split dataset into 90% train / 10% test for honest evaluation

    model_p = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',    # ------------> handles unequal priority class sizes
        C=5,                        # ------------> tighter decision boundary for better separation
        solver='lbfgs'              # ------------> best solver for multiclass TF-IDF problems
    ).fit(X_train, y_train_p)       # ------------> to train priority model on train split only for honest accuracy

    model_c = LogisticRegression(
        max_iter=2000,
        C=5,
        solver='lbfgs'
    ).fit(X, df['category_enc'])    # ------------> to train category model on full data for strongest predictions

    model_t = LinearRegression().fit(X, df['resolution_time'])      # ------------> to train resolution time regression model on full data

    accuracy = accuracy_score(y_test_p, model_p.predict(X_test))    # ------------> to calculate honest accuracy on unseen test set
    cm       = confusion_matrix(y_test_p, model_p.predict(X_test))  # ------------> to compute confusion matrix on unseen test set

    return tfidf, X, model_p, model_c, model_t, accuracy, cm

# -------------------------
# SUBJECT PREDICTION
# -------------------------
def predict_subject(user_text, tfidf, X, df):   # ------------> to find the most similar ticket subject from the dataset using cosine similarity on a 5000-row sample for speed

    # Use positional indices to avoid IndexError after filtering/reset
    sample_pos = np.random.RandomState(42).choice(len(df), size=5000, replace=False)
    X_sample   = X[sample_pos]
    df_sample  = df.iloc[sample_pos].reset_index(drop=True)

    user_vector = tfidf.transform([clean_text(user_text)])
    similarity  = cosine_similarity(user_vector, X_sample)
    index       = similarity.argmax()

    return df_sample.iloc[index]['subject']

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Ticket AI", layout="centered")
st.title("🎯 Customer Support Ticket Classification System")

df, le_p, le_c = load_data()
tfidf, X, mp, mc, mt, accuracy, cm = train(df)


text = st.text_area("✍️ Enter Customer Complaint")

if text.strip():

    x = tfidf.transform([clean_text(text)])

    # Predictions
    p = le_p.inverse_transform([mp.predict(x)[0]])[0]               #---> to predict the priority level and inverse transform back to original label
    c = le_c.inverse_transform([mc.predict(x)[0]])[0]               #---> to predict the issue category and inverse transform back to original label
    s = predict_subject(text, tfidf, X, df)                         #---> to find the most similar ticket subject using cosine similarity
    t = max(1, round(float(mt.predict(x)[0]), 2))                   #---> to predict resolution time, minimum capped at 1 hour
        # Show model performance in sidebar so it doesn't clutter the main UI
    st.subheader("📊 Model Performance")
    st.write("Accuracy:", round(accuracy, 2))       #---> to display the accuracy of the priority model in the sidebar

    if t <= 2:
        message = "Quick Resolution Expected"
    elif t <= 24:
        message = "Moderate Resolution Time"
    else:
        message = "May Take Longer Than Usual"

    # 🔥 UI OUTPUT
    st.markdown("## 🚨 Priority Level")
    st.error(p.upper())                                     #---> to display the predicted priority level formatted as an error message in uppercase

    st.markdown("### 🏷️ Issue Category")
    st.info(c)                                              #---> to display the predicted issue category as an informational message

    st.markdown("### 📝 Product")
    st.success(s)                                            #---> to display the most similar product/subject as a success message
    st.error("Product category may not be perfect some times")

    st.markdown("### ⏱️ Estimated Resolution Time")
    st.success(f"**{t} hours**")                            #---> to display the predicted resolution time in bold
    st.info(message)                                        #---> to display a human-readable resolution time message

else:
    st.info("Enter complaint to categorize the ticket")
