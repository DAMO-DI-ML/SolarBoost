import numpy as np
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error as MSE
# from scipy.optimize import minimize
#from scipy.linalg import lstsq
# from numpy.linalg import lstsq
# from scipy.optimize import nnls
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import lsq_linear
import lightgbm as lgb
# from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.ar_model import AutoReg
# from statsmodels.tsa.arima.model import ARIMA
import pickle
import logging
from util import KalmanFilter, create_matrix_vectorized, RCR

   
class MM_dstb:
    def __init__(self, n_estimator=100, max_depth=3, boost_interval=10, fit_cap=True, lambda_1=1e3):
        self.n_estimator = n_estimator #迭代最大步数
        self.max_depth = max_depth
        self.trees = []
        self.cap_models = []
        self.end_threshold = 0.001 #停止条件
        self.loss = []
        self.loss_capacity = []
        self.boost_interval = boost_interval #每个基学习器拟合几棵树
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
        # C [t,1] 假设一个seq内装机不变
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
            return np.sum(predictions*c_i,axis=2)
    
    def custom_mse(self, yi, yi_pred):
        """自定义均方误差损失函数"""
        yi = yi.reshape(self.t, self.seq, self.k)
        yi_pred = yi_pred.reshape(self.t, self.seq, self.k)
        residual = np.sum(yi,axis=2, keepdims=True) - np.sum(self.cap*yi_pred, axis=2, keepdims=True)  # 计算残差
        grad = -2 * residual*self.cap # 梯度
        hess = np.repeat(2 *self.cap**2, self.seq, axis=1)  # Hessian
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

    # def optimize_cap_KalmanFilter(self, residuals, residuals_pred):
    #     #这里有个是要调一个优化方法，逐个时间点优化
    #     constraints = {'type': 'eq', 'fun': lambda ci: np.sum(ci) - 1}
    #     bounds = [(0,1)]*self.k
    #     loss_capacity = 0
    #     residuals,residuals_pred  = residuals.reshape(-1,self.seq,self.k), residuals_pred.reshape(-1,self.seq,self.k)
    #     i = 0
    #     while i < self.t:
    #         def objective(c):
    #             loss = self.up_bound(c,i)
    #             return loss
    #         loss_capacity += objective(self.cap[i,:,:])
    #         if self.k > self.seq:
    #             result = minimize(objective, self.cap[i,:,:].flatten(), constraints=constraints, bounds = bounds, method='SLSQP')
    #         else:
    #             result = minimize(objective, self.cap[i,:,:].flatten(), bounds = bounds, method='SLSQP')
    #         if result.success == False or np.isnan(result.x).any():
    #             logging.info(f"minimize False for t {i}")
    #             if np.isnan(self.cap[i-1,:,:]).any() or i == 0:
    #                 self.cap[i,:,:] = np.ones((1, 1, self.k))/ self.k
    #             else:
    #                 self.cap[i,:,:] = self.cap[i-1,:,:]
    #         else:
    #             self.cap[i,:,:] = result.x.reshape(1,1,self.k)
    #         self.norm_ci(i)
    #         i += 1
    #     #self.KalmanFilter_cap()
            
    def KalmanFilter_cap(self, r=0.001, p=0.9):
        A = np.eye(self.k)  # 状态转移矩阵
        H = np.eye(self.k)  # 观测矩阵
        #Q = np.array([q]*self.k)  # 过程噪声协方差
        Q = (create_matrix_vectorized(self.k, p))*r**2
        R = np.array([r**2]*self.k)  # 观测噪声协方差

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
            #model = ARIMA(data, order=(1, 0, 1))
            model = AutoReg(data, lags=1)
            model= model.fit()
            #第3步之后的预测值都是一样的？
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
        #预测装机，先假设和最后t时刻装机是一样的
        cap_pred = []
        for i in self.k:
            cap_pred.append(self.cap_models.predict(start=self.t, end=self.t+t-1))
        return np.row_stack(cap_pred).reshape(t,1,self.k)
    

def generate_data(t=300, seq=96, k=15, d=3,seed=42):
    def AR(x0, n, phi, sigma):
    # 假设是1阶AR
        X = np.zeros(n)
        X[0] = x0  # 设置初始值
        epsilon = np.random.normal(loc=0, scale=sigma, size=n)
        # 生成数据
        for t in range(1, n):
            X[t] = phi * X[t-1] + epsilon[t]
        return X
    np.random.seed(seed)
    X = np.random.rand(t, seq, k, d)
    #X = np.row_stack((X,X))

    #c_i = np.row_stack((np.array([1,2]*t),np.array([2,1]*t)))
    ci = np.column_stack([AR(x, t, phi, 0.0001) for x, phi in zip(np.random.rand(2*k)[k:],np.random.normal(loc=1, scale=0.0001, size=k))])
    # ci = np.random.rand(k).reshape(-1,k)
    # ci = np.repeat(ci, t, axis=0)
    ci = ci.reshape(t,1,k)
    C = ci.sum(axis=2, keepdims=True)

    y = np.sin(X[:,:,:,0]) + X[:,:,:,1] + X[:,:,:,2]**2 #+ np.random.normal(loc=0, scale=0.001, size=(t, k))
    y = (y*ci).sum(axis=2,keepdims=True)
    return X, y, ci, C


def plot_ci(ci, k, xlabel = 'grid', ylabel = 'time step', title = 'capacity true'):
    sns.heatmap(ci.reshape(-1,k), annot=False, fmt=".1f", cmap='viridis')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

if __name__ == '__main__':
    # from ec_data import reduce_size
    from pathlib import Path
    import os
    import pandas as pd
    import logging
    import argparse
    import datetime
    import re
    import itertools
    
    parser = argparse.ArgumentParser(description='[MM model] Train MM model')
    parser.add_argument('--city', type=str, default='dongying')
    parser.add_argument('--plant_type', type=str, default='dstb_low_pv')
    parser.add_argument('--data_dir', type=str, default='./data')
    # parser.add_argument('--start_date', type=str, default='20230801')
    # parser.add_argument('--end_date', type=str, default='20240929')
    parser.add_argument('--train_day', type=int, default=None)
    parser.add_argument('--test_day', type=int, default=30)
    parser.add_argument('--resolution', type=float, default=0.2)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--boost_interval', type=int, default=1)
    parser.add_argument('--optimize_cap_methed', type=str, default='KalmanFilter',
                        choices=['KalmanFilter', 'l2', 'None'])
    parser.add_argument('--optimize_cap_backward', type=bool, default=False)
    
    args = parser.parse_args()

    city = args.city
    plant_type = args.plant_type
    data_dir = args.data_dir
    # start_date = args.start_date
    # end_date = args.end_date
    test_day = args.test_day
    train_day = args.train_day
    resolution = args.resolution
    n_estimator = args.n_estimators
    max_depth = args.max_depth
    boost_interval = args.boost_interval
    optimize_cap_methed = args.optimize_cap_methed
    dataset_dir = os.path.join(data_dir,'')
    log_dir = os.path.join(data_dir,'logs/model/')
    model_dir = os.path.join(data_dir,'model/')
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
                                       mode='a')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[file_handler, stream_handler])
                        
    data_path = os.path.join(dataset_dir,f'{resolution}.npz')
    
    try:
        data = np.load(data_path)
        X, y, dates, grids = data['X'], data['y'], data['dates'], data['grids']
        logging.info(f'read data from {data_path}')
    except:
        logging.error(f'read data from {data_path} failed')
        
    grid_num = len(grids)
    if train_day is not None:
        train_day = min(train_day,len(dates)-test_day)
    else:
        train_day = len(dates)-test_day
    
    train_X, train_y, test_X, test_y = X[len(dates) - train_day -test_day:-test_day,:,:,:], y[len(dates) - train_day -test_day:-test_day,:,:], X[-test_day:,:,:,:], y[-test_day:,:,:]
    
    logging.info(f"train_data range from: {str(dates[len(dates) - train_day])[:10]} to {str(dates[-test_day-1])[:10]},contain {train_day} days")
    logging.info(f"test_data range from: {str(dates[-test_day])[:10]} to {str(dates[-1])[:10]},contain {test_day} days")
    logging.info(f"resolution is: {resolution}, grid_num is {grid_num}")

    model_path = os.path.join(model_dir,f'{train_day}_{test_day}_{resolution}_{n_estimator}_{max_depth}_{boost_interval}.pkl')

    logging.info(f'MM train start!')
    model = MM_dstb(n_estimator=n_estimator, max_depth=max_depth, boost_interval=boost_interval, fit_cap=True)
    model.fit(train_X, train_y)
    logging.info(f'MM train end!')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f'save MM model in {model_path}')
    train_y_pred = model.predict(train_X)
    test_y_pred = model.predict(test_X)
    logging.info(f'MM predict end!')
    
    RCR_train = RCR(train_y.flatten(),np.sum((train_y_pred.reshape(-1,96,grid_num)*model.cap),axis=2).flatten(),1)
    RCR_test = RCR(test_y.flatten(),np.sum((test_y_pred.reshape(-1,96,grid_num)*model.cap[-test_day:,:]),axis=2).flatten(),1)
    logging.info(f'MM score train RCR:{RCR_train}, test RCR: {RCR_test}')

    train_X_flatten,test_X_flatten = train_X.reshape(train_day*96,-1), test_X.reshape(test_day*96,-1)
    train_X_mean,test_X_mean = np.mean(train_X,axis=2).reshape(train_day*96,-1), np.mean(test_X,axis=2).reshape(test_day*96,-1)

    logging.info(f'flatten lgm train and predict!')
    train_data = lgb.Dataset(train_X_flatten, label=train_y.flatten())
    trees = lgb.train({
        'objective': 'regression',
        'max_depth': max_depth,
        'learning_rate': 0.01,
        'verbosity': -1,
        'n_estimators': n_estimator}, train_data)

    train_Y_pred_flatten= trees.predict(train_X_flatten)
    test_Y_pred_flatten = trees.predict(test_X_flatten)
    RCR_train_flatten = RCR(train_y.flatten(),train_Y_pred_flatten,1)
    RCR_test_flatten = RCR(test_y.flatten(),test_Y_pred_flatten,1)
    logging.info(f'flatten score train RCR:{RCR_train_flatten}, test RCR: {RCR_test_flatten}')

    logging.info(f'mean lgm train and predict!')
    train_data = lgb.Dataset(train_X_mean, label=train_y.flatten())
    trees = lgb.train({
        'objective': 'regression',
        'max_depth': max_depth,
        'learning_rate': 0.01,
        'verbosity': -1,
        'n_estimator': n_estimator}, train_data)

    train_Y_pred_mean = trees.predict(train_X_mean)
    test_Y_pred_mean = trees.predict(test_X_mean)

    RCR_train_mean = RCR(train_y.flatten(),train_Y_pred_mean,1)
    RCR_test_mean = RCR(test_y.flatten(),test_Y_pred_mean,1)
    logging.info(f'mean score train RCR:{RCR_train_mean}, test RCR: {RCR_test_mean}')



    