import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ==============================
# Load Dataset
# ==============================
data = pd.read_csv("mental_health.csv")

# ==============================
# Text Cleaning
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

data["text"] = data["text"].apply(clean_text)

# ==============================
# Exploratory Data Analysis (EDA)
# ==============================

# Label distribution
sns.countplot(x="label", data=data)
plt.title("Sentiment Distribution")
plt.show()

# ==============================
# Feature Engineering
# ==============================
X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ==============================
# Model 1: Logistic Regression
# ==============================
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# ==============================
# Model 2: Naive Bayes
# ==============================
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

# ==============================
# Model Comparison
# ==============================
print("Logistic Regression Accuracy:", lr_acc)
print("Naive Bayes Accuracy:", nb_acc)

# Confusion Matrix (Best Model)
best_model = lr if lr_acc >= nb_acc else nb
best_name = "Logistic Regression" if best_model == lr else "Naive Bayes"

cm = confusion_matrix(y_test, best_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt="d")
plt.title(f"Confusion Matrix - {best_name}")
plt.show()

print("Best Model Selected:", best_name)

# ==============================
# Save Best Model
# ==============================
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))