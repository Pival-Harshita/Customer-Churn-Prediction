# Customer Churn Prediction using Machine Learning and Ensemble Learning

## Project Overview

Customer churn prediction helps identify customers who are likely to discontinue a company's services. Early prediction enables businesses to improve customer retention and reduce revenue loss.

This project implements a customer churn prediction pipeline using data preprocessing, clustering techniques, multiple machine learning classifiers, and ensemble learning methods to compare predictive performance.

---

## Features

- Data acquisition
- Data preprocessing
- Feature engineering
- Clustering techniques
- Single machine learning classifiers
- Ensemble learning models
- Model evaluation and comparison

---

## Machine Learning Models

### Clustering
- K-Means
- K-Medoids
- X-Means
- Random Clustering

### Single Classification Models
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Naive Bayes
- Gradient Boosting
- Artificial Neural Network (ANN)

### Ensemble Models
- Voting Classifier
- Bagging
- AdaBoost
- Stacking

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- XGBoost
- Imbalanced-Learn

---

## Project Structure

```text
Customer-Churn-Prediction/

│── clustering/
│── data_acquisition/
│── data_preprocessing/
│── dataset/
│── ensemble_classification/
│── single_classification/
│── README.md
│── requirements.txt
│── .gitignore
```

---

## Dataset

This project uses the **Telco Customer Churn Dataset**.

The dataset is **not included** in this repository.

Download it from:

https://github.com/Geo-y20/Telco-Customer-Churn-/blob/main/Telco%20Customer%20Churn.csv

After downloading, place the CSV file inside:

```text
dataset/
    Telco_Customer_Churn.csv
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Pival-Harshita/Customer-Churn-Prediction
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the required Python script from the corresponding module:

- `data_preprocessing/preprocessing.py`
- `clustering/main.py`
- `single_classification/main.py`
- `ensemble_classification/main.py`

---

## Results

The project compares the performance of multiple clustering approaches, single classifiers, and ensemble learning models for telecom customer churn prediction.

---

## Future Improvements

- Hyperparameter optimization
- Deep learning models
- Explainable AI (SHAP/LIME)
- Deployment using Streamlit or Flask

---

## Author

**Pival Harshita**