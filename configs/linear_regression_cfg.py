from easydict import EasyDict

cfg = EasyDict()
cfg.dataframe_path = 'linear_regression_dataset.csv'
cfg.base_functions = None


def init_base_functions(count: int):
    cfg.base_functions = [lambda x: x ** i for i in range(1, count + 1)]
