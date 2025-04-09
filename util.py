import numpy as np
import os
import pickle

            
def RCR(ytrue, ypred, cap, threshold=0):
    # def RCR(ytrue, ypred, cap, threshold=0.2):
    if ytrue.size != ypred.size:
        raise ValueError('Incompatible size!')
    elif ytrue.size == 0:
        return 0
    else:
        ytrue_compare = ytrue.copy()
        idx = ytrue_compare >= cap * threshold
        if ytrue[idx].shape[0] == 0:
            return 1
        else: 
            return 1 - np.sqrt(np.mean(((ytrue[idx] - ypred[idx]) / cap) ** 2))
            

def create_matrix_vectorized(size, p):
    # 创建行索引数组
    row_indices = np.arange(size).reshape(-1, 1)
    # 创建列索引数组
    col_indices = np.arange(size).reshape(1, -1)
    # 利用广播机制计算绝对差
    abs_diff = np.abs(row_indices - col_indices)
    # 应用公式得到最终矩阵
    result = p ** abs_diff
    return result

class KalmanFilter:
    def __init__(self, A, H, Q, R, B=0, P=None, x0=None):
        """
        初始化函数
        :param A: 状态转移矩阵
        :param H: 观测矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 观测噪声协方差
        :param B: 控制输入矩阵，默认为0
        :param P: 初始估计误差协方差
        :param x0: 初始状态估计
        """
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B
        self.x = x0 if x0 is not None else np.ones(A.shape[0])/A.shape[0]
        self.P = P if P is not None else np.eye(A.shape[0])

    def predict(self, u=0):
        """
        预测步骤
        :param u: 外部控制输入
        """
        # 预测新状态
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        # 更新估计误差协方差
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        """
        更新步骤
        :param z: 测量值
        """
        # 计算卡尔曼增益
        K = np.dot(np.dot(self.P, self.H.T), 
                   np.linalg.pinv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R))
        # 根据测量更新状态
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        # 更新估计误差协方差
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
    def get_state(self):
        return self.x

    def simulate(self,steps):
        """
        初始化函数
        :param A: 状态转移矩阵
        :param H: 观测矩阵
        :param Q: 过程噪声协方差
        :param R: 观测噪声协方差
        :param B: 控制输入矩阵，默认为0
        :param P: 初始估计误差协方差
        :param x0: 初始状态估计
        """
        u=0
        measurements = []
        x = self.x.copy()
        for _ in range(steps):
            # 预测步骤
            u = 0
            w = np.random.multivariate_normal(mean=[0]*len(self.Q), cov=self.Q)  # 过程噪声
            x = self.A @ x + self.B * u + w
            
            # 更新测量
            v = np.random.normal(0, np.sqrt(self.R),size = len(x))  # 测量噪声
            z = self.H @ x + v
            
            measurements.append(z)
        return measurements

def load_or_train_model(model_path, train_func, *args, **kwargs):
    """Load existing model if available, otherwise train new one"""
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded existing model from {model_path}")
        return model
    
    print(f"Training new model...")
    model = train_func(*args, **kwargs)
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved new model to {model_path}")
    return model

#画图函数