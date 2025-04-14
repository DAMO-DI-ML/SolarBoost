import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import lsq_linear
import lightgbm as lgb
from statsmodels.tsa.ar_model import AutoReg
import pickle
import logging
from util import KalmanFilter, create_matrix_vectorized, RCR
import os

   
class MM_dstb:
    def __init__(self, n_estimator=100, max_depth=3, boost_interval=10, fit_cap=True, lambda_1=1e3):
        self.n_estimator = n_estimator
        self.max_depth = max_depth
        self.trees = []
        self.cap_models = []
        self.end_threshold = 0.001
        self.loss = []
        self.loss_capacity = []
        self.boost_interval = boost_interval
        self.learning_rate = 0.01
        self.fit_cap = fit_cap
        self.X=None
        self.y=None
        self.cap=None
        self.lambda_1 = lambda_1
        self.history_cap = []

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        
    def generate_yi(self):
        return self.y*np.ones((self.t, self.seq, self.k))/ self.k
    
    def init_ci(self,capacity_init):
        if capacity_init is not None:
            self.cap = np.vstack([capacity_init]*self.t).reshape(self.t, 1, self.k)
        else:
            self.cap = np.ones((self.t, 1, self.k))/ self.k
    
    def fit(self, X, y, capacity_init=None):
        # x [t,sqe,k,d]
        # y [t,sqe,1]
        # C [t,1]
        self.t, self.seq, self.k, self.d = X.shape
        self.X, self.y = X, y
        self.init_ci(capacity_init)
         
        # Pre-reshape X to avoid repeated reshaping in loop
        X_reshaped = self.X.reshape(-1, self.d)
        yi = self.generate_yi()
        residuals = self.y.copy()
        estimated_previous = np.zeros(yi.shape)
        
        # Pre-configure LightGBM parameters
        gbm_params = {
            'objective': self.custom_mse,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.boost_interval,
            'verbosity': -1,
            'force_row_wise': True  # Can speed up training
        }
        
        for i in range(self.n_estimator//self.boost_interval):
            residuals = np.repeat(residuals, self.k, axis=2)/self.k
            gbm = lgb.LGBMRegressor(**gbm_params)
            gbm.fit(X_reshaped, residuals.flatten())
            self.trees.append(gbm)
            
            # Calculate predictions and reshape in one step
            estimated_delta = gbm.predict(X_reshaped).reshape(self.t, self.seq, self.k)
            
            if self.fit_cap:
                self.update_cap(estimated_previous, estimated_delta)

            estimated_previous += estimated_delta
            # Optimize the residuals calculation
            residuals = self.y - np.sum(estimated_previous * self.cap, axis=2, keepdims=True)
            self.loss.append(np.linalg.norm(residuals))
            self.history_cap.append(self.cap.copy())

            if (i+1)%10 == 0:
                logging.info(f'n_estimator:{i+1},loss:{self.loss[-1]},capacity min:{np.min(self.cap)},capacity max:{np.max(self.cap)}')
            
    def predict_trees(self, X, n_estimator=None):
        if X.ndim > 2:
            X = X.reshape(-1, self.d)
        predictions = np.zeros(len(X))
        if n_estimator is None:
            n_estimator = len(self.trees)
        else:
            n_estimator = min(n_estimator,len(self.trees))
        for tree in self.trees[:n_estimator]:
            predictions += tree.predict(X)
        return predictions.reshape(-1,1)

    def predict(self, X, ci=None):
        if X.ndim > 2:
            X = X.reshape(-1, self.d)
        predictions = self.predict_trees(X)
        if ci is None:
            return predictions
        else:
            return np.sum(predictions*ci,axis=2)
    
    def custom_mse(self, yi, yi_pred):
        yi = yi.reshape(self.t, self.seq, self.k)
        yi_pred = yi_pred.reshape(self.t, self.seq, self.k)
        residual = np.sum(yi,axis=2, keepdims=True) - np.sum(self.cap*yi_pred, axis=2, keepdims=True)
        grad = -2 * residual*self.cap
        hess = np.repeat(2 *self.cap**2, self.seq, axis=1)
        return grad.flatten(), hess.flatten()


    def update_cap(self, estimated_previous, estimated_delta):
        alpha = np.zeros((self.t, self.k))
        
        # Pre-compute common values
        sum_y = np.sum(self.y, axis=1)
        
        for i in range(self.t):
            q_t = estimated_previous[i,:,:].sum(axis=0).reshape(-1,1)
            delta_t = estimated_delta[i,:,:].sum(axis=0).reshape(-1,1)
            dq = delta_t @ q_t.T  # Using @ for matrix multiplication
            
            # Optimize matrix calculations
            if i == 0 and len(self.history_cap)==0:
                mu = sum_y[i]*(delta_t+q_t)
                Sigma = self.k * np.diag(np.sum(estimated_delta[i,:,:]**2,axis=0)) + q_t @ q_t.T + dq + dq.T
            elif i==0 and len(self.history_cap)>0:
                mu = sum_y[i]*(delta_t+q_t) + self.lambda_1*self.history_cap[-1][i,:,:].reshape(-1,1)
                Sigma = self.k * np.diag(np.sum(estimated_delta[i,:,:]**2,axis=0) + self.lambda_1/self.k) + q_t @ q_t.T + dq + dq.T
            else:
                mu = sum_y[i]*(delta_t+q_t) + self.lambda_1*alpha[i-1,:].reshape(-1,1)
                Sigma = self.k * np.diag(np.sum(estimated_delta[i,:,:]**2,axis=0) + self.lambda_1/self.k) + q_t @ q_t.T + dq + dq.T
            
            try:
                result = lsq_linear(Sigma, mu.flatten(), bounds=(0,1))
                alpha[i,:] = result.x
            except:
                logging.info(f'time {i} error: SVD did not converge in Linear Least Squares')
            
        self.cap = np.maximum(alpha,0).reshape(self.cap.shape)

    def KalmanFilter_cap(self, r=0.001, p=0.9):
        A = np.eye(self.k)  # State transition matrix
        H = np.eye(self.k)  # Observation matrix
        Q = (create_matrix_vectorized(self.k, p))*r**2
        R = np.array([r**2]*self.k)  

        kf = KalmanFilter(A, H, Q, R)

        for i in range(self.t):
            z = self.cap[i,:,:].flatten()
            kf.predict()
            kf.update(z)
            self.cap[i,:,:] = (kf.get_state()).reshape(self.cap[i,:,:].shape)
            self.cap[i,:,:] = np.clip(self.cap[i,:,:], 0, 1)
        
        
    def smooth_ci(self, lag=1):
        if len(self.cap_models) == 0:
            self.cap_models = [None]*self.k
        for i in range(self.k):
            data = self.cap[:,:,i].flatten()
            model = AutoReg(data, lags=1)
            model= model.fit()
            predictions = model.predict(start=1, end=self.t-1)
            self.cap[:,:,i] = np.concatenate((self.cap[0,:,i].reshape(1,1),predictions.reshape(-1,1)),axis=0)
            self.cap_models[i] = model
    
    def norm_ci(self,i=None):
        if i:
            self.cap[i,:,:] = np.clip(self.cap[i,:,:], 0, 1)
            self.cap[i,:,:] = self.cap[i,:,:]/np.sum(self.cap[i,:,:], axis=1)
            if np.isnan(self.cap[i,:,:]).any():
                self.cap[i,:,:] = np.ones(self.cap[i,:,:].shape)/ self.k
        else:
            self.cap = np.clip(self.cap, 0, 1)
            self.cap = self.cap/np.sum(self.cap, axis=2, keepdims=True)
        
        
    def predict_cap(self,t):
        cap_pred = []
        for i in self.k:
            cap_pred.append(self.cap_models.predict(start=self.t, end=self.t+t-1))
        return np.row_stack(cap_pred).reshape(t,1,self.k)
    
def AR(x0, n, phi, sigma):

    X = np.zeros(n)
    X[0] = x0  
    epsilon = np.random.normal(loc=0, scale=sigma, size=n)

    for t in range(1, n):
        X[t] = phi * X[t-1] + epsilon[t]
    return X
    



def plot_ci(ci, k, xlabel = 'grid', ylabel = 'time step', title = 'capacity true'):
    sns.heatmap(ci.reshape(-1,k), annot=False, fmt=".1f", cmap='viridis')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

if __name__ == '__main__':
    import argparse
    import datetime
    import os
    
    parser = argparse.ArgumentParser(description='[MM model] Train MM model')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--train_day', type=int, default=None)
    parser.add_argument('--test_day', type=int, default=30)
    parser.add_argument('--resolution', type=float, default=0.2)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--boost_interval', type=int, default=1)
    
    args = parser.parse_args()

    # Setup logging
    log_dir = os.path.join(args.data_dir, 'logs/model/')
    model_dir = os.path.join(args.data_dir, 'model/')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
    ]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # Load data
    data_path = os.path.join(args.data_dir, f'{args.resolution}.npz')
    try:
        data = np.load(data_path)
        X, y, dates, grids = data['X'], data['y'], data['dates'], data['grids']
        logging.info(f'Read data from {data_path}')
    except:
        logging.error(f'Failed to read data from {data_path}')
        raise

    # Prepare train/test split
    grid_num = len(grids)
    train_day = min(args.train_day, len(dates)-args.test_day) if args.train_day else len(dates)-args.test_day
    
    train_X = X[len(dates) - train_day - args.test_day:-args.test_day,:,:,:]
    train_y = y[len(dates) - train_day - args.test_day:-args.test_day,:,:]
    test_X = X[-args.test_day:,:,:,:]
    test_y = y[-args.test_day:,:,:]

    # Train model
    logging.info('Training MM model...')
    model = MM_dstb(n_estimator=args.n_estimators, 
                    max_depth=args.max_depth, 
                    boost_interval=args.boost_interval, 
                    fit_cap=True)
    model.fit(train_X, train_y)

    # Save model
    model_path = os.path.join(model_dir, 
                             f'{train_day}_{args.test_day}_{args.resolution}_' \
                             f'{args.n_estimators}_{args.max_depth}_{args.boost_interval}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f'Saved model to {model_path}')

    # Evaluate
    train_y_pred = model.predict(train_X)
    test_y_pred = model.predict(test_X)
    
    train_score = RCR(train_y.flatten(), 
                      np.sum((train_y_pred.reshape(-1,96,grid_num)*model.cap),axis=2).flatten(), 
                      1)
    test_score = RCR(test_y.flatten(),
                     np.sum((test_y_pred.reshape(-1,96,grid_num)*model.cap[-args.test_day:,:]),axis=2).flatten(),
                     1)
    
    logging.info(f'MM scores - Train RCR: {train_score:.4f}, Test RCR: {test_score:.4f}')


def train_and_evaluate_model(X, y, ci, C, yi, train_size=280, test_size=None, n_estimator=100, max_depth=3, lambda_1=1,boost_interval = 5):
    """Train and evaluate the boosting model with specified parameters"""
    # Split data
    t,seq,k = X.shape[:3]

    train_X = X[:train_size,:,:,:] if test_size is None else X[:-test_size,:,:,:]
    train_y = y[:train_size,:,:] if test_size is None else y[:-test_size,:,:]
    train_C = C[:train_size,:,:] if C is not None else None
    train_ci = ci[:train_size,:,:] if ci is not None else None
    test_X = X[train_size:,:,:,:] if test_size is None else X[-test_size:,:,:,:]
    test_y = y[train_size:,:,:] if test_size is None else y[-test_size:,:,:]
    test_C = C[train_size:,:,:] if C is not None else None
    test_ci = ci[train_size:,:,:] if ci is not None else None
    test_yi = yi[train_size:,:,:] if yi is not None else None
    # Initialize and train model
    boosting_regressor = MM_dstb(n_estimator=n_estimator, 
                                max_depth=max_depth, 
                                boost_interval=boost_interval, 
                                fit_cap=True, 
                                lambda_1=lambda_1)
    train_y_norm = train_y/train_C if train_C is not None else train_y
    boosting_regressor.fit(train_X, train_y_norm)
    
    # Generate predictions
    test_y_pred = boosting_regressor.predict(test_X)
    pred_ci = boosting_regressor.cap.reshape(-1,k)/np.sum(boosting_regressor.cap.reshape(-1,k),axis=1,keepdims=True)
    return {'model':boosting_regressor, 
            'pred_y': test_y_pred, 
            'test_y': test_y, 
            'train_ci': train_ci, 
            'train_C': train_C, 
            'pred_ci': pred_ci, 
            'test_C': test_C, 
            'test_ci': test_ci,
            'test_yi': test_yi,
            't': t,
            'seq': seq,
            'k': k}

def plot_results(boosting_regressor, train_ci, train_C, k = 15):
    """Plot capacity comparison between true and predicted values"""
    fig = plt.figure(figsize=(13, 5))
    
    # Plot true capacity
    plt.subplot(1,2,1)
    plot_ci((train_ci/train_C), k)
    
    # Plot predicted capacity
    capacity_pred = boosting_regressor.cap.reshape(-1,k)/np.sum(boosting_regressor.cap.reshape(-1,k),axis=1,keepdims=True)
    plt.subplot(1,2,2)
    plot_ci(capacity_pred, k, title='capacity pred')
    
    plt.tight_layout()
    return fig


def load_or_save_experiment(config):
    """Load or run experiment based on configuration
    
    Args:
        config: Dictionary containing:
            - name: Name of model ('ar1' or 'kalman')
            - model_params: Dictionary of model parameters
            - data_params: Dictionary of data generation parameters including t, seq, k, d
    """
    # Model paths
    model_params = config['model_params']
    data_params = config['data_params']
    model_path = f'./models/{config["name"]}_model_n{model_params["n_estimator"]}_l{model_params["lambda_1"]}_t{model_params["train_size"]}.pkl'
    results_path = model_path.replace('.pkl', '_results.pkl')
    
    # Get dimensions from config
    t,seq,k = None,None,None
    if 'city' not in config['name']:
        t, seq, k = data_params['t'], data_params['seq'], data_params['k']
    
    # Load or train
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f"Loaded existing {config['name']} model from {model_path}")
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file {results_path} does not exist")
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded existing {config['name']} results from {results_path}")
    else:
        X, y, ci, C, yi = None, None, None, None, None
        # Generate data based on model type
        if config['name'] == 'kalman':
            from kalman import generate_kalman_data
            X, y, ci, C, yi = generate_kalman_data(**data_params)
        elif config['name'] == 'ar1':
            from ar1 import generate_ar_data
            X, y, ci, C, yi = generate_ar_data(**data_params)
        elif config['name'] == 'city_a':
            from city_a import load_city_a_data
            X, y, dates, grids = load_city_a_data(data_params)
            t,seq,k = X.shape[:3]
        
        # Train new model
        results = train_and_evaluate_model(X, y, ci, C, yi,**model_params)
        model = results['model']
        
        # Save model and results
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved new {config['name']} model to {model_path}")

        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved {config['name']} results to {results_path}")
    
    return model, results, t, seq, k

def calculate_aggregate_rmse(results, model, t, seq, k, train_size, first = False):
    """Calculate RMSE for aggregate output"""
    if t is None:
        t = results['t']
    if seq is None:
        seq = results['seq']
    if k is None:
        k = results['k']
    if train_size is None:
        train_size = t-results['test_y'].shape[0]
    if results['test_C'] is None:
        results['test_C'] = 1
    if first:
        pred_sum = (results['pred_y'].reshape(-1, seq, k) * model.cap[train_size-t:,:])[:,:,0]
        true_norm = (results['test_yi']/results['test_C'])[:,:,0]
    else:
        pred_sum = np.sum((results['pred_y'].reshape(-1, seq, k) * model.cap[train_size-t:,:]), axis=2).flatten()
        true_norm = (results['test_y']/results['test_C']).flatten()
    return np.sqrt(np.mean((pred_sum - true_norm) ** 2))

def calculate_capacity_rmse(results, k):
    capacity_pred = results['model'].cap.reshape(-1,k)/np.sum(results['model'].cap.reshape(-1,k),axis=1,keepdims=True)
    capacity_true = (results['train_ci']/results['train_C']).reshape(-1,k)
    return np.sqrt(np.mean((capacity_pred - capacity_true) ** 2))