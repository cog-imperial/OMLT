import numpy as np
import pandas as pd
from random import random

n_samples = 10000
w = 5

x = np.linspace(-2,2,n_samples)    
df = pd.DataFrame(x, columns=['x'])
df['y'] = np.sin(w*x) + x**2 + np.array([np.random.uniform()*0.1 for _ in range(n_samples)])

plt.plot(df['x'],df['y'])
plt.show()

df.to_csv("sin_quadratic.csv")