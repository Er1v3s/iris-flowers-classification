import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix

plt.ion()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

IMAGES_PATH = Path() / "../images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id: object, tight_layout: object = True,
             fig_extension: object = "png", resolution: object = 300) -> object:
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


iris = datasets.load_iris()
iris.keys()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

print(df.describe())

# Box plots for all features
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris['feature_names']):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='target', y=feature, data=df)
    plt.title(feature)
save_fig("BoxPlot")

# Histograms of residuals for all features
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris['feature_names']):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(feature)
save_fig("Residuals")

# 3D scatterplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df[iris['feature_names'][0]],
           df[iris['feature_names'][1]],
           df[iris['feature_names'][2]],
           c=df['target'])
ax.set_xlabel(iris['feature_names'][0])
ax.set_ylabel(iris['feature_names'][1])
ax.set_zlabel(iris['feature_names'][2])
plt.title('3D scatterplot')
save_fig("3DScatterplot")

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(method='pearson'), annot=True, cmap='coolwarm')
plt.title('Correlation matrix')
save_fig("CorrelationMatrix")

# Split data learning/testing
X = df[iris['feature_names']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)

def calculate_specificity(conf_matrix):
    TN = np.diag(conf_matrix)
    FP = conf_matrix.sum(axis=0) - TN
    specificity = TN / (TN + FP)
    return specificity.mean()

k_values = [20, 40, 60]

for k in k_values:
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)

    knn_predictions = knn_clf.predict(X_test)

    # Confusion matrix
    knn_conf_matrix = confusion_matrix(y_test, knn_predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    knn_cmd = ConfusionMatrixDisplay(knn_conf_matrix, display_labels=iris['target_names'])
    knn_cmd.plot(ax=ax)
    plt.title(f'Confusion matrix for KNN k={k}')
    save_fig(f"KNN_ConfusionMatrix_k{k}")

    print(f"\nClassification report for KNN k={k}:\n")

    accuracy_knn = accuracy_score(y_test, knn_predictions)
    f1_knn = f1_score(y_test, knn_predictions, average='macro')
    specificity_knn = calculate_specificity(knn_conf_matrix)
    recall_knn = recall_score(y_test, knn_predictions, average='macro')

    results_knn_df = pd.DataFrame({
        "Metrics": ["Accuracy", "F1 score", "Specificity", "Recall"],
        "Values": [accuracy_knn, f1_knn, specificity_knn, recall_knn]
    })

    print(results_knn_df)

plt.show(block=True)