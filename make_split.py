import numpy as np

random_gen = np.random.default_rng(seed = 42)
index = random_gen.permutation(25000)
train, test = index[:20000], index[20000:]
np.savetxt("split_train.csv", train, fmt="%d")
np.savetxt("split_test.csv", test, fmt="%d")

