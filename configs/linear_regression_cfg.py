from easydict import EasyDict
import numpy as np
cfg = EasyDict()
cfg.dataframe_path = 'linear_regression_dataset.csv'

cfg.base_functions = [lambda x: x ** i for i in range(1, 3 + 1)]
