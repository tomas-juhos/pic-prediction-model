import lightgbm
from scipy.stats import randint
import pandas as pd


class LightGBM:

    model_gbm = None
    params = None
    params_list = None
    max_grid_iter = None

    x_train = None
    y_train = None

    x_valid = None
    x_test = None

    y_train_pred = None
    y_pred = None

    def __init__(self):
        self.name = 'LightGBM'

        default_params = {'max_depth': 5,
                          'num_leaves': 10,
                          'min_data_in_leaf': 100,
                          'seed': 5, 'TIME_STEPS': 12, 'verbose': 0}

        self.useMinMax = 0
        self.set_params(default_params)

    def set_params(self, params):
        self.params = params

    def generate_random_params(self, N):
        self.params_list = []
        for i in range(N):
            self.params_list.append({'max_depth': randint.rvs(1, 20),
                                     'num_leaves': randint.rvs(2, 30),
                                     'min_data_in_leaf': randint.rvs(1, 20),
                                     'TIME_STEPS': randint.rvs(5, 50),
                                     'seed': 5, 'verbose': -1
                                     })
        self.max_grid_iter = N
        return self.params_list

    def train(self):
        # todo feed dataset correctly
        train_data = lightgbm.Dataset(
            pd.DataFrame(self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1] * self.x_train.shape[2])),
            label=pd.DataFrame(self.y_train[:, -1, :]))
        self.model_gbm = lightgbm.train(self.params, train_data, early_stopping_rounds=0)

    def predict_valid(self):
        self.y_train_pred = self.model_gbm.predict(self.y_train.reshape(self.x_valid.shape[0], self.y_train.shape[1]))

    def predict(self):
        self.y_pred = self.model_gbm.predict(
            self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1] * self.x_test.shape[2]))

    # todo implement as
