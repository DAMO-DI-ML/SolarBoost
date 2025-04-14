import numpy as np
import os
import logging
import datetime
from func import *

print('a')
CITY_A_CONFIG = {
        'name': 'city_a',
        'data_params': {
        'data_dir': './data',
        'resolution': 0.2,},
        'model_params': {
        'train_size': None,
        'test_size': 30,
        'n_estimator': 1000,
        'max_depth': 3,
        'boost_interval': 1,
        'lambda_1': 10000
        }
    }
def setup_logging(data_dir):
    """Setup logging configuration"""
    log_dir = os.path.join(data_dir, 'logs/model/')
    model_dir = os.path.join(data_dir, 'model/')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, f'city_a_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'))
    ]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return model_dir

def load_city_a_data(params):
    """Load data from npz file"""
    data_path = os.path.join(params['data_dir'], f'{params["resolution"]}.npz')
    try:
        
        data = np.load(data_path)
        X, y, dates, grids = data['X'], data['y'], data['dates'], data['grids']
        logging.info(f'Read data from {data_path}')
        return X, y, dates, grids
    except:
        logging.error(f'Failed to read data from {data_path}')
        raise
    

def run_city_experiment(goal = ['rmse_y']):
    """Run City A experiment and return results"""
    model, results, t, seq, k = load_or_save_experiment(CITY_A_CONFIG)
    # Setup logging
    # setup_logging(params['data_dir'])
    rmse_y = None
    if 'rmse_y' in goal:
        rmse_y = calculate_aggregate_rmse(results, model, t, seq, k, CITY_A_CONFIG['model_params']['train_size'])
    results = {'rmse_y': rmse_y}
    return results
    # # Load data
    # data_path = os.path.join(params['data_dir'], f'{params["resolution"]}.npz')
    # X, y, dates, grids = load_city_a_data(data_path)
    
    # model_path = f'./models/city_a_model_r{params["resolution"]}_n{params["n_estimators"]}.pkl'
    # model = load_or_train_model(model_path, MM_dstb, **params)
    # predictions = model.predict(X)
    
    # # Calculate RMSE for aggregate output
    # rmse = np.sqrt(np.mean((predictions.sum(axis=2) - y.sum(axis=2)) ** 2))
    # return rmse 