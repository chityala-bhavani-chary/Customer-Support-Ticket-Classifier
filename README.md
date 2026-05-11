# 🎯 Customer Support Ticket Classification System

An NLP + Machine Learning based web application that automatically classifies customer support tickets into:

- 🚨 Priority Level
- 🏷️ Issue Category
- 📝 Product / Ticket Subject
- ⏱️ Estimated Resolution Time

Built using Python, Scikit-learn, NLP, and Streamlit.

---

## 🚀 Live Demo

🔗 Streamlit App  
https://customer-support-ticket-classifier.streamlit.app/

🔗 GitHub Repository  
https://github.com/chityala-bhavani-chary/Customer-Support-Ticket-Classifier

---

# 📌 Project Overview

This project simulates a real-world customer support ticketing system using Machine Learning and Natural Language Processing.

The model analyzes customer complaints and predicts:

- Priority Level
- Issue Category
- Related Product
- Estimated Resolution Time

The system helps automate ticket routing and prioritization for support teams.

---

# 🧠 Workflow

## 1. Data Preprocessing

- Lowercasing
- Regex Cleaning
- Stopword Removal
- Text Normalization

---

## 2. Feature Engineering

TF-IDF Vectorization is used to convert text into numerical vectors.

```python
TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,2),
    stop_words='english',
    sublinear_tf=True,
    min_df=2
)
````

---

## 3. Models Used

| Task                       | Model               |
| -------------------------- | ------------------- |
| Priority Prediction        | Logistic Regression |
| Category Prediction        | Logistic Regression |
| Product Prediction         | Cosine Similarity   |
| Resolution Time Prediction | Linear Regression   |

---

# 📊 Features

✅ Customer Complaint Classification
✅ Priority Prediction
✅ Product Similarity Matching
✅ Resolution Time Estimation
✅ NLP Text Cleaning
✅ Interactive Streamlit UI
✅ Real-world ML Pipeline

---

# 🖥️ Example

## Input

```python
"My payment failed and money got deducted"
```

## Output

```python
Priority Level: HIGH
Issue Category: Billing
Product: Payment Gateway
Estimated Resolution Time: 4.5 hours
```

---

# 📂 Dataset Used

### CFPB Complaints Dataset

[https://www.kaggle.com/datasets/raulenriquez/cfpb-complaints?select=cfpb_complaints_2019_to_2022](https://www.kaggle.com/datasets/raulenriquez/cfpb-complaints?select=cfpb_complaints_2019_to_2022)

### Multilingual Customer Support Tickets

[https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)

### Customer Support Tickets Dataset (200K Records)

[https://www.kaggle.com/datasets/mirzayasirabdullah07/customer-support-tickets-dataset-200k-records](https://www.kaggle.com/datasets/mirzayasirabdullah07/customer-support-tickets-dataset-200k-records)

### Have used the all the sets combined into a single csv file :
The project uses a real-world customer support ticket dataset containing issue descriptions, priorities, categories, products, and resolution times.

Dataset File in  in this repository:

🔗 [customer_support_tickets_200k.csv](./customer_support_tickets_200k.csv)

OR

🔗 [customer_support_tickets_200k.zip](./customer_support_tickets_200k.zip)

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/chityala-bhavani-chary/Customer-Support-Ticket-Classifier.git
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run ML_Project.py
```

---

# 📁 Project Structure

```bash
Customer-Support-Ticket-Classifier/
│
├── ML_Project.py
├── requirements.txt
├── customer_support_tickets_200k.zip
└── README.md
```

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Streamlit

---

# 📈 Future Improvements

* Confusion Matrix Visualization
* ROC Curve
* Deep Learning Models (LSTM)
* Transformer Models (BERT)
* Better Subject Prediction
* Cloud Optimization

---

# 👨‍💻 Author

## Bhavani Chary

MBA Student | Data Analyst | ML & NLP Enthusiast

### Skills

* Python
* SQL
* Power BI
* Machine Learning
* NLP
* Streamlit

---

# ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub.
