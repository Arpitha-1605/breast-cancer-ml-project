# ================================
# IMPORT LIBRARIES
# ================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ================================
# LOAD DATASET
# ================================
data = load_breast_cancer()
X = data.data
y = data.target

# ================================
# TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ================================
# APPLY PCA
# ================================
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# ================================
# APPLY LDA
# ================================
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# ================================
# MODELS
# ================================
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(max_depth=4),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(max_depth=4)
}

# ================================
# EVALUATION FUNCTION
# ================================
def evaluate_models(X_train, X_test, y_train, y_test, title):
    print(f"\n========== {title} ==========")

    acc_list = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n{name}")
        print("Accuracy:", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall:", round(rec, 4))
        print("F1 Score:", round(f1, 4))

        acc_list.append(acc)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Malignant", "Benign"],
                    yticklabels=["Malignant", "Benign"],
                    cbar=False)

        plt.title(f"{title} - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    return acc_list


# ================================
# RUN ALL THREE CASES
# ================================
acc_original = evaluate_models(X_train, X_test, y_train, y_test, "Original Data")
acc_pca = evaluate_models(X_train_pca, X_test_pca, y_train, y_test, "PCA Data")
acc_lda = evaluate_models(X_train_lda, X_test_lda, y_train, y_test, "LDA Data")

# ================================
# INTERACTIVE COMPARISON GRAPH
# ================================
model_names = list(models.keys())

# Prepare data for plotting
import pandas as pd

df = pd.DataFrame({
    "Model": model_names * 3,
    "Accuracy": [round(a,3) for a in acc_original] +
                [round(a,3) for a in acc_pca] +
                [round(a,3) for a in acc_lda],
    "Method": ["Original"]*5 + ["PCA"]*5 + ["LDA"]*5
})

fig = px.bar(
    df,
    x="Model",
    y="Accuracy",
    color="Method",
    barmode="group",
    title="Model Accuracy Comparison (Original vs PCA vs LDA)"
)

fig.update_traces(text=df["Accuracy"], textposition='outside')
fig.update_layout(title_x=0.5, yaxis=dict(range=[0.8,1]))

fig.show()

# ================================
# UNSUPERVISED LEARNING (K-MEANS)
# ================================
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

print("\nK-Means Silhouette Score:", round(silhouette_score(X, clusters), 4))

plt.figure(figsize=(5,4))
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()
