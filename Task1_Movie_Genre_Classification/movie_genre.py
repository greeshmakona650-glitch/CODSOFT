import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# LOAD DATASET

data = pd.read_csv(
    "train_data.txt",
    sep=" ::: ",
    engine="python",
    header=None,
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

print("Dataset Shape:", data.shape)
# FEATURES & TARGET

X = data["DESCRIPTION"]
y = data["GENRE"]
# TF-IDF VECTORIZATION

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = tfidf.fit_transform(X)
# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# LOGISTIC REGRESSION

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)

print("\n===== Logistic Regression =====")
print("Accuracy:", lr_acc)
# NAIVE BAYES
nb = MultinomialNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)

print("\n===== Naive Bayes =====")
print("Accuracy:", nb_acc)
# SUPPORT VECTOR MACHINE
svm = LinearSVC()

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)

print("\n===== SVM =====")
print("Accuracy:", svm_acc)
# BEST MODEL
results = {
    "Logistic Regression": lr_acc,
    "Naive Bayes": nb_acc,
    "SVM": svm_acc
}

best_model = max(results, key=results.get)

print("\n🏆 Best Model:", best_model)

# USER INPUT PREDICTION


movie_plot = input("\n🎬 Enter Movie Plot:\n")

movie_plot_tfidf = tfidf.transform([movie_plot])

prediction = lr.predict(movie_plot_tfidf)

print("\n🎯 Predicted Genre:", prediction[0])