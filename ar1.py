import numpy as np
import matplotlib.pyplot as plt
from func import *

# AR1 experiment configuration
AR1_CONFIG = {
    'name': 'ar1',
    'model_params': {
        'n_estimator': 1000,
        'lambda_1': 1,
        'train_size': 280
    },
    'data_params': {
        't': 300,
        'seq': 96,
        'k': 15,
        'd': 3
    }
}

def AR(x0, n, phi, sigma):
    """Generate AR(1) process"""
    X = np.zeros(n)
    X[0] = x0  # Initial value
    epsilon = np.random.normal(loc=0, scale=sigma, size=n)
    for t in range(1, n):
        X[t] = phi * X[t-1] + epsilon[t]
    return X

def generate_ar_data(t=300, seq=96, k=15, d=3, seed=42):
    """Generate synthetic data for AR1 experiment"""
    np.random.seed(seed)
    
    # Generate feature matrix
    X = np.random.rand(t, seq, k, d)
    
    # Generate capacity values using AR(1) process
    ci = np.column_stack([
        AR(x, t, phi, 0.0001) 
        for x, phi in zip(
            np.random.rand(2*k)[k:],
            np.random.normal(loc=1, scale=0.0001, size=k)
        )
    ])
    ci = ci.reshape(t, 1, k)
    C = ci.sum(axis=2, keepdims=True)

    # Generate target values
    yi = np.sin(X[:,:,:,0]) + X[:,:,:,1] + X[:,:,:,2]**2
    y = (yi*ci).sum(axis=2, keepdims=True)
    
    return X, y, ci, C, yi

def run_ar1_experiment(goal = ['fig']):
    """Run AR1 experiment and return results"""
    # Load or run experiment
    model, results, t, seq, k = load_or_save_experiment(AR1_CONFIG)
    
    # Create plots
    fig, rmse_y, rmse_y1, rmse_c, output,output_min,output_max,output_mean = None, None, None, None, None, None, None, None
    if 'fig' in goal:
        fig = plot_results(model, results['train_ci'], results['train_C'])
    if 'rmse_y' in goal:
        rmse_y = calculate_aggregate_rmse(results, model, t, seq, k, AR1_CONFIG['model_params']['train_size'])
    if 'rmse_y1' in goal:
        rmse_y1 = calculate_aggregate_rmse(results, model, t, seq, k, AR1_CONFIG['model_params']['train_size'], first = True)
    if 'rmse_c' in goal:
        rmse_c = calculate_capacity_rmse(results, k)
    if 'output' in goal:
        output = results['pred_y'].reshape(-1, seq, k)[:,:,0]
        output_min = output.min()
        output_max = output.max()
        output_mean = output.mean()
    results = {'fig': fig, 'rmse_y': rmse_y, 'rmse_y1': rmse_y1, 'rmse_c': rmse_c, 
               'output_min': output_min, 'output_max': output_max, 'output_mean': output_mean}
    return results

