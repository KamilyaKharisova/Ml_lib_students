from easydict import EasyDict

cfg = EasyDict()
cfg.dataframe_path = 'linear_regression_dataset.csv'

"""list of basis functions"""
cfg.base_functions = [lambda x, arg=i: pow(x, arg) for i in range(8 + 1)]