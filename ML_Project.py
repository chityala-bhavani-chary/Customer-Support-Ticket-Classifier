import pandas as pd, streamlit as st, re, nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Setup & Cleaning ---
@st.cache_resource
def setup():
    nltk.download('stopwords')
    return set(nltk.corpus.stopwords.words('english'))

STOPWORDS = setup()

def clean(t):
    t = re.sub(r'[^a-z ]', '', str(t).lower())
    return " ".join([w for w in t.split() if w not in STOPWORDS and len(w) > 2])

# --- Data & Training ---
@st.cache_data
def get_model():
    df = pd.read_csv("customer_support_tickets.csv").rename(columns=lambda x: x.strip().lower())
    cols = {'ticket_description':'text', 'issue_category':'cat', 'priority_level':'prior', 'resolution_time_hours':'time'}
    df = df.rename(columns=cols)[list(cols.values())].dropna()
    df['clean'] = df['text'].apply(clean)

    # Higher accuracy config: LinearSVC with balanced weights
    # pipe = TF-IDF + Classifier
    def create_pipe(clf):
        return make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=5000), clf)

    X_train, X_test, y_train, y_test = train_test_split(df['clean'], df['prior'], test_size=0.2)
    
    # Priority Model (LinearSVC is usually more accurate for text than LogReg)
    m_p = create_pipe(LinearSVC(class_weight='balanced', C=0.5))
    m_p.fit(X_train, y_train)
    
    # Category and Time Models
    m_c = create_pipe(LinearSVC(class_weight='balanced'))
    m_c.fit(df['clean'], df['cat'])
    
    m_t = create_pipe(Ridge()) # Ridge is more robust for small-scale regression
    m_t.fit(df['clean'], df['time'])
    
    acc = accuracy_score(y_test, m_p.predict(X_test))
    return m_p, m_c, m_t, acc

# --- UI ---
st.set_page_config(page_title="AI Ticket", layout="wide")
st.title("🎯 Smart Ticket Pro")

m_p, m_c, m_t, acc = get_model()
text = st.text_area("✍️ Customer Complaint:")

if text.strip():
    cleaned = clean(text)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Priority", m_p.predict([cleaned])[0])
    col2.metric("Category", m_c.predict([cleaned])[0])
    col3.metric("Est. Time", f"{round(m_t.predict([cleaned])[0], 1)}h")
    
    st.progress(acc)
    st.caption(f"Current Model Accuracy: {acc:.2%}")
else:
    st.info("Awaiting input...")
