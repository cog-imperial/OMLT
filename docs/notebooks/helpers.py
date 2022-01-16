import random
import numpy as np


### we define a simple black-box function to generate data
class SimpleCat:
    def __init__(self):
        self.cat_idx = [3]
        self.name = 'simple_cat'

    def get_bounds(self):
        temp_bounds = [(-2.0, 2.0) for _ in range(3)]

        # lightgbm requires categorical variables as integers
        # in this case 0 is not meant to be 'smaller' than one
        # but the integers are rather keys to individual categories
        temp_bounds.append([0, 1])
        return temp_bounds

    def __call__(self, X):
        cat = X[-1]
        X = np.asarray_chkfinite(X[:-1])
        X0 = X[:-1]
        X1 = X[1:]

        add1 = X0[0]
        add2 = X1[0]

        if cat == 0:
            return 6 * (add1 + add2)
        elif cat == 1:
            return (add1 + add2) ** 2


def generate_samples(num_samples, bb_func, bb_bnds):
    data = {
        'X': [],
        'y': []
    }

    for _ in range(num_samples):
        sample = []

        # iterate through all dimension bounds
        for idx, var_bnds in enumerate(bb_bnds):

            if idx not in bb_func.cat_idx:
                # pick uniformly random number between continuous bounds
                # for continuous (conti) variables
                val = random.uniform(var_bnds[0], var_bnds[1])
            else:
                # pick random integer and map it to string for categorical (cat) variables
                cat_int = random.randrange(0, len(var_bnds))
                val = cat_int

            # populate the sample
            sample.append(val)

        data['X'].append(sample)
        data['y'].append(bb_func(sample))
    return data


def generate_gbt_data():
    bb_func = SimpleCat()
    bb_bnds = bb_func.get_bounds()
    data = generate_samples(100, bb_func, bb_bnds)
    return data, bb_func.cat_idx