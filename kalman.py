import numpy as np
import matplotlib.pyplot as plt
from func import *
from util import KalmanFilter, create_matrix_vectorized, load_or_train_model
import os
import pickle

KALMAN_CONFIG = {
    'name': 'kalman',
    'model_params': {
        'n_estimator': 25,
        'lambda_1': 10,
        'train_size': 280
    },
    'data_params': {
        't': 300,
        'seq': 96,
        'k': 15,
        'd': 3,
        'seed': 42,
        'p': 0.9,
        'r': 0.001
    }
}

def generate_kalman_data(t=300, seq=96, k=15, d=3, seed=42, p=0.9, r=0.001):
    """Generate synthetic data using Kalman filter"""
    np.random.seed(seed)
    X = np.random.rand(t, seq, k, d)
    
    A = np.eye(k)  # State transition matrix
    H = np.eye(k)  # Observation matrix
    Q = (create_matrix_vectorized(k, p))*r**2  # Process noise covariance
    R = np.array([r**2]*k)  # Observation noise covariance
    x0 = np.random.rand(k)  # Initial state
    
    kf = KalmanFilter(A, H, Q, R, x0=x0)
    ci = kf.simulate(t)
    ci = np.array(ci).reshape(t, 1, k)
    C = ci.sum(axis=2, keepdims=True)
    
    # Generate target values
    y = np.sin(X[:,:,:,0]) + X[:,:,:,1] + X[:,:,:,2]**2
    y = (y*ci).sum(axis=2, keepdims=True)
    
    return X, y, ci, C

def run_kalman_experiment():
    """Run Kalman experiment and return results"""
    # Load or run experiment
    model, results, t, seq, k = load_or_save_experiment(KALMAN_CONFIG)
    
    # Create plots
    fig = plot_results(model, results['train_ci'], results['train_C'])
    
    print('mode is aggregate')
    rmse = calculate_aggregate_rmse(results, model, t, seq, k, KALMAN_CONFIG['model_params']['train_size'])
    return rmse, fig

# # Generate data
# X, y, ci, C = generate_kalman_data()

# # Train and evaluate model
# boosting_regressor, train_ci, train_C = train_and_evaluate_model(X, y, ci, C, lambda_1=10, n_estimator=25)

