from easydict import EasyDict
import numpy as np
cfg = EasyDict()
cfg.dataframe_path = 'linear_regression_dataset.csv'

var1 =  [lambda x: pow(x, i) for i in range(1,9)]
var2 =  [lambda x: pow(x, i) for i in range(1,2)]
var3 =  [lambda x: pow(x, i) for i in range(1,101)]

cfg.base_functions =  var1 # range(1,101)