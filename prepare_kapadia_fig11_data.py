import os
import numpy as np


d = np.load("kapadia_experiment/test/metadata/1.npy", allow_pickle=True, encoding="latin1")
dlist = [d]
for idx in range(4):
    nd = np.copy(d)
    nd[1] = "sample_{}.png".format(180 - (1 + idx))
    nd[2] = "{}".format(180 - (1 + idx))
    dlist.append(nd.tolist())
dlist = np.asarray(dlist)
os.remove("kapadia_experiment/test/metadata/1.npy")
np.save("kapadia_experiment/test/metadata/1.npy", dlist)


