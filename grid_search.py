import numpy as np
import itertools

# grid search
param_grid = {
    "lr": [0.01, 0.001, 0.0001],
    "temperature": [0.8, 0.9, 1],
    'weight_decay': [0.001, 0.0001, 0.0005],
    'augmentation_warm_up_epoches': [0, 40, 80],
    'cf_weight': [0.1, 0.2, 0.3],
    'augment_threshold': [4, 12, 20]
}
hyperparams = []
for values in itertools.product(*param_grid.values()):
    point = dict(zip(param_grid.keys(), values))
    hyperparams.append(point)

# print(hyperparams)
for i in hyperparams:
    print(i)
