EXPERIMENT_CONFIGS = {
    'kalman': {
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
    },
    'ar1': {
        'name': 'ar1',
        'model_params': {
            'n_estimator': 25,
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
} 