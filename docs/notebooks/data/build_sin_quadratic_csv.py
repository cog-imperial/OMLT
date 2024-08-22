import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n_samples = 10000
w = 5

x = np.linspace(-2, 2, n_samples)
rng = np.random.default_rng()
sin_quads = pd.DataFrame(x, columns=["x"])
sin_quads["y"] = (
    np.sin(w * x) + x**2 + np.array([rng.uniform() * 0.1 for _ in range(n_samples)])
)

plt.plot(sin_quads["x"], sin_quads["y"])
plt.show()

sin_quads.to_csv("sin_quadratic.csv")
