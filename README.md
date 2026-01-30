# Bank-Customer-Churn-Prediction-System


![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive machine learning web application that predicts customer churn probability using 6 different ML models with an intuitive Flask-based interface.

Live Demo: [https://bank-churn-predictions.onrender.com](live soon)

Presentation: [View Full Presentation](Uploading Soon)



Table of Contents

- [Features](#features)
- [Technologies](#technologies)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Key Findings](#key-findings)
- [Screenshots](#screenshots)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)


Features

Machine Learning Models
6 Trained Models: Logistic Regression, Random Forest, SVM (with SMOTE), XGBoost, Deep Learning (Neural Network), KNN
Ensemble Predictions: Get predictions from all models with average probability
SMOTE Balancing: Advanced technique to handle class imbalance
High Accuracy: ROC-AUC scores above 0.85

User Interface
Responsive Design: Works on desktop, tablet, and mobile
Interactive Forms: User-friendly input validation
Visual Results: Charts and graphs for easy interpretation
Risk Assessment: Color-coded risk levels with recommendations
Real-time Feedback: Instant prediction results

Technical Features
RESTful API: Programmatic access to predictions
Model Persistence: Pre-trained models loaded efficiently
Error Handling: Robust error management
Scalable Architecture: Easy to add new models

Technologies

#Backend
Python 3.10
Flask 2.3- Web framework
Scikit-learn 1.2 - Machine learning
TensorFlow 2.12 - Deep learning
XGBoost 1.7 - Gradient boosting
Pandas & NumPy - Data processing
Joblib - Model serialization

#Frontend
HTML5 & CSS3
Bootstrap 5.3 - UI framework
JavaScript (ES6+)
Chart.js - Data visualization
Font Awesome - Icons

#Deployment
Gunicorn- WSGI server
Render.com - Cloud platform
Git - Version control
