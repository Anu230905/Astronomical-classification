# Astronomical-classification

Astronomical Object Classification
An end-to-end Data Science and Machine Learning project implemented in a Jupyter Notebook to classify astronomical objects (such as stars, galaxies, and quasars) using photometric survey data. The project benchmarks multiple machine learning algorithms—including XGBoost, Random Forest, SVM, Neural Networks (TensorFlow), and PCA for dimensionality reduction—to identify celestial bodies based on spectral and spatial attributes.

Table of Contents
Overview

Tech Stack

Notebook Workflow

Repository Structure

Installation & Setup

Running the Notebook

Models & Evaluation

License

Overview
Modern astronomical surveys (e.g., SDSS) capture vast quantities of observational data. Manual inspection of every celestial body is impossible, making automated machine learning classification essential. This project preprocesses astronomical feature sets (e.g., sky angles, photometric filter magnitudes u,g,r,i,z, and redshift), scales features using MinMaxScaler, reduces dimensionality with PCA, and compares several machine learning classifiers and deep learning models.

Tech Stack
Environment: Jupyter Notebook (%matplotlib inline)

Data Handling: pandas, numpy

Data Visualization: matplotlib, seaborn

Machine Learning (Scikit-Learn & XGBoost):

Preprocessing & Scaling: LabelEncoder, MinMaxScaler

Dimensionality Reduction: PCA (Principal Component Analysis)

Validation: train_test_split, cross_val_predict

Classifiers: XGBClassifier, RandomForestClassifier, SVC, KNeighborsClassifier, GaussianNB, SGDClassifier

Evaluation Metrics: confusion_matrix, precision_score, recall_score, f1_score

Deep Learning: tensorflow

Utilities: time, warnings

Notebook Workflow
The analysis is structured sequentially inside the notebook (astronomical_object_detection.ipynb):

Environment Setup & Data Loading: Installs xgboost via !pip install xgboost, configures plot styles with seaborn, suppresses deprecation warnings, and loads survey data into pandas.

Exploratory Data Analysis (EDA): Visualizes filter band correlations, redshift distributions, and class frequencies (e.g., STAR vs. GALAXY vs. QSO).

Data Preprocessing & Transformation:

Encodes categorical targets via LabelEncoder.

Normalizes continuous observational features using MinMaxScaler.

Applies PCA (Principal Component Analysis) for feature reduction and visual projection.

Model Training & Benchmarking:

Trains multiple classical classifiers (RandomForestClassifier, XGBClassifier, SVC, KNeighborsClassifier, GaussianNB, SGDClassifier).

Builds a Neural Network model using tensorflow.

Measures training and execution speed using time.

Model Evaluation:

Uses cross_val_predict for out-of-fold predictions.

Evaluates models using multi-class metrics: Precision, Recall, F1-Score, and Confusion Matrices.

Repository Structure
Plaintext
astronomical-object-detection/
├── astronomical_object_detection.ipynb   # Single Jupyter Notebook with full analysis & models
├── data/
│   └── sky_server_astronomy_data.csv    # Photometric survey dataset
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
Installation & Setup
Clone the repository:

Bash
git clone https://github.com/your-username/astronomical-object-detection.git
cd astronomical-object-detection
Create and activate a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Running the Notebook
Launch Jupyter directly from your terminal:

Bash
jupyter notebook astronomical_object_detection.ipynb
Or using JupyterLab:

Bash
jupyter lab
Execute the cells sequentially (Cell -> Run All). The notebook will install xgboost if not already present, run feature scaling and PCA, train all classifiers, and plot comparative performance charts.

Models & Evaluation
The notebook evaluates and compares model performance across multiple metrics:

Algorithm	Precision	Recall	F1-Score	Execution Time
XGBoost	High	High	High	Fast
Random Forest	High	High	High	Fast
Support Vector Machine (SVC)	Medium	Medium	Medium	Moderate
K-Nearest Neighbors (KNN)	Medium	Medium	Medium	Fast
Gaussian Naive Bayes	Baseline	Baseline	Baseline	Very Fast
SGD Classifier	Baseline	Baseline	Baseline	Very Fast
TensorFlow Neural Network	High	High	High	Training Heavy
License
Distributed under the MIT License. See LICENSE for details.
