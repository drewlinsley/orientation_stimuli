import numpy as np
import sys
from matplotlib import pyplot as plt


f = sys.argv[2]
name = sys.argv[1]

data = np.load(f)
# plt.imshow(data)
plt.plot(data)
plt.show()

