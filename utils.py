import streamlit as st
import numpy as np
import pandas as pd

# This is a helper script for the Streamlit app
# It contains the utility functions for prediction

def preprocess_input(input_data, scaler):
    """
    Preprocess the input data for prediction
    
    Args:
        input_data: numpy array of input features
        scaler: trained StandardScaler object
        
    Returns:
        Scaled input data
    """
    return scaler.transform(input_data)

def get_risk_level(probability):
    """
    Determine risk level based on probability
    
    Args:
        probability: Probability of having diabetes
        
    Returns:
        Risk level and recommendation
    """
    if probability < 0.3:
        return "Low", "Continue maintaining a healthy lifestyle."
    elif probability < 0.7:
        return "Moderate", "Consider consulting with a healthcare professional."
    else:
        return "High", "Please consult with a healthcare professional as soon as possible."

def format_feature_importance(features, coefficients):
    """
    Format feature importance for display
    
    Args:
        features: List of feature names
        coefficients: List of coefficients from logistic regression
        
    Returns:
        DataFrame with feature importance
    """
    return pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(coefficients)
    }).sort_values(by='Importance', ascending=False)
