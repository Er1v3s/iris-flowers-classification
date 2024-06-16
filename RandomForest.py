import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix

plt.ion()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

IMAGES_PATH = Path() / "images"
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

# Classifier Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=False)
rf_clf.fit(X_train, y_train)
predictions = rf_clf.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(8, 6))
cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=iris['target_names'])
cmd.plot(ax=ax)
plt.title('Confusion matrix')
save_fig("RF_ConfussionMatrix")

print(f"\nClassification report for Random Forest:\n")

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')

TP = np.diag(conf_matrix)
FP = conf_matrix.sum(axis=0) - TP
FN = conf_matrix.sum(axis=1) - TP
TN = conf_matrix.sum() - (FP + FN + TP)

specificity = TN / (TN + FP)
specificity = specificity.mean()

results_df = pd.DataFrame({
    "Metrics": ["Accuracy", "F1 Score", "Specificity", "Recall"],
    "Values": [accuracy, f1, specificity, recall]
})

print(results_df)
plt.show(block=True)
