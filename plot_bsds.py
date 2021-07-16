import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# Need to change to bootstrapped distributions! But for cosyne just plot perf.
df = pd.read_csv("cosyne.csv")
df = df.sort_values("human_variance_explained")[::-1]

# axes = df.plot.barh(color={"#1342ba", "#858ea6"}, figsize=(5, 7))
axes = df.plot.bar(color={"#1342ba"}, figsize=(5, 7))
plt.ylim([0, 1])
axes.legend(bbox_to_anchor=(0, 1), loc='center right', ncol=1)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
# plt.tight_layout()
plt.show()
plt.close("all")

