import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from func import *

X, y, ci, C = generate_data()
t, seq, k, d = X.shape
s = 280
train_X, train_y, train_C, train_ci = X[:s,:,:,:], y[:s,:,:], C[:s,:,:], ci[:s,:,:]
test_X, test_y, test_C, test_ci = X[s:,:,:,:], y[s:,:,:], C[s:,:,:], ci[s:,:,:]

boosting_regressor = MM_dstb(n_estimator=100, 
                             max_depth=3, 
                             boost_interval=5, 
                             fit_cap=True, 
                             lambda_1=1)
#boosting_regressor.fit(train_X, train_y/train_C, train_ci[0,:]/train_C[0,:].reshape(-1,1))
boosting_regressor.fit(train_X, train_y/train_C)
train_y_pred = boosting_regressor.predict(train_X)
test_y_pred = boosting_regressor.predict(test_X)


plt.figure(figsize=(13, 5))
plt.subplot(1,2,1)
plot_ci((train_ci/train_C),k)  # annot=True 在每个方块上显示数值

capacity_pred = boosting_regressor.history_cap[-1].reshape(-1,k)/np.sum(boosting_regressor.cap.reshape(-1,k),axis=1,keepdims=True)
plt.subplot(1,2,2)
plot_ci(capacity_pred,k, title = 'capacity pred')
# plt.savefig(dpi=120,fname='./results/figure/capacity_simulation_ar1.png')
plt.show()