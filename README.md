ML-Powered Employee Analytics Dashboard
This project provides a comprehensive solution for predicting employee attrition and performance ratings using a Streamlit-based web application. The dashboard is powered by machine learning models trained on the Employee-Attrition.csv dataset.

Model Training (model_build.py): This script cleans the data, preprocesses it using pipelines (including a custom Winsorization function), and trains two separate Logistic Regression models for attrition and performance prediction. The trained models are saved as performance_model.pkl and attrition_model.pkl.

Interactive Web Dashboard (mlapp.py): This Streamlit application loads the trained models and presents a user-friendly interface. Users can input various employee data points and receive real-time predictions for both attrition and performance rating. The application is designed with two distinct tabs, making it easy to switch between prediction tasks.

Key Features
Two-in-One Prediction: Predict both employee attrition (Yes/No) and performance rating (1-4) from a single interface.

Data Preprocessing Pipeline: The models are built using a robust pipeline that includes:

Winsorization: A custom function to handle outliers by capping values at the 5th and 95th percentiles.

log transform : to perform the outlier handling

Standard Scaling: To normalize numerical features.

One-Hot Encoding: To convert categorical features into a numerical format suitable for the models.

Interactive Streamlit UI: The web application is built with Streamlit, providing a simple, responsive, and intuitive way to interact with the models without any coding knowledge.

Scalable Architecture: The separation of the model training and serving code allows for easy updates and maintenance.

Project Structure
Employee-Attrition.csv: The raw dataset used for training the models.

model_build.py: Python script for data preprocessing, model training, and serialization.

attrition_model.pkl: The serialized machine learning model for predicting employee attrition.

performance_model.pkl: The serialized machine learning model for predicting employee performance rating.

mlapp.py: The Streamlit web application script for the interactive dashboard.
