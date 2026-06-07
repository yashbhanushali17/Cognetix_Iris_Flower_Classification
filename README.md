# Iris Flower Classification

## Project Description

This project was developed as part of the Cognetix Machine Learning Internship Foundation Task.

The objective of this project is to classify Iris flowers into three different species using machine learning. The prediction is made based on four flower measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The three species present in the dataset are:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

In this project, I trained and compared three different machine learning algorithms:

* Logistic Regression
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)

Their performance was evaluated using accuracy score, confusion matrix, precision, recall, and F1-score.

---

## Steps Followed

### 1. Data Loading

* Loaded the Iris dataset using Pandas.
* Added column names manually.
* Checked dataset shape and basic information.

### 2. Exploratory Data Analysis (EDA)

* Checked class distribution of flower species.
* Created a pairplot to understand relationships between features.
* Generated a correlation heatmap to see feature correlations.
* Verified that there were no missing values in the dataset.

### 3. Data Preprocessing

* Encoded species names into numerical labels using LabelEncoder.
* Split the dataset into training and testing sets using an 80:20 ratio.

### 4. Model Training

The following models were trained:

* Logistic Regression
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)

### 5. Model Evaluation

Each model was evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

### 6. Visualizations

The following visualizations were created:

* Pairplot
* Correlation Heatmap
* Class Distribution Chart
* Confusion Matrix Heatmaps
* Decision Tree Feature Importance Graph

### 7. Model Saving

The best-performing model was saved using Joblib as:

`iris_best_model.pkl`

---

## How to Run

### Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Run the Project

```bash
python model.py
```

---

## Files Generated

* pairplot.png
* correlation_heatmap.png
* class_distribution.png
* logistic_model_confusion_matrix.png
* decision_tree_model_confusion_matrix.png
* knn_model_confusion_matrix.png
* decision_tree_feature_importance.png
* iris_best_model.pkl

---

## Conclusion

This project helped me understand the complete machine learning workflow, including data exploration, preprocessing, model training, evaluation, visualization, and model saving. The Iris dataset is small and well-structured, which allowed all three models to achieve excellent classification performance.
