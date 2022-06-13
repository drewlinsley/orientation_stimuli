import numpy as np


d = np.load("gilbert_angelluci_flanker_kinoshita/test/metadata/1.npy", allow_pickle=True, encoding="latin1")
d = d.reshape(-1, 12)
d[0, 1] = "sample_0.png"
d[0, 2] = "0"
np.save("gilbert_angelluci_flanker_kinoshita/test/metadata/1.npy", d)

