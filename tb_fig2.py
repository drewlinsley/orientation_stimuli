import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


no_surround = np.load('plaid_no_surround_outputs_data.npy')
surround = np.load('orientation_tilt_outputs_data.npy')

stride = np.floor(180 / surround.shape[-1]).astype(int)
# thetas = {0: -90, 30//5: -60, 60//5: -30, 90//5: 0, 120//5: 30, 150//5: 60}
thetas = {
    0 // stride: -90,
    30 // stride: -60,
    60 // stride: -30,
    90 // stride: 0,
    120 // stride: 30,
    150 // stride: 60
}

ranges = np.arange(-90, 90, stride)
ranges = ranges[:surround.shape[1]]

print(no_surround.shape)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
f = plt.figure(figsize=(13, 2))
plt.suptitle("Trott and Born Fig 2")
for idx, (theta, label) in enumerate(thetas.items()):
    theta = int(theta)
    ax = plt.subplot(1, len(thetas), idx + 1)

    it_surround = surround[theta]
    it_no_surround = no_surround[90]
    plt.plot(ranges, it_surround, label="surround", color="Red", alpha=0.5, marker=".")  # noqa
    plt.plot(ranges, it_no_surround, label="no surround", color="Black", alpha=0.5, marker=".")  # noqa

    # Draw verticle lines
    plt.axvline(x=ranges[np.argmax(it_surround)], color="Red", linestyle="--", alpha=0.5)  # noqa
    plt.axvline(x=ranges[np.argmax(it_no_surround)], color="Black", linestyle="--", alpha=0.5)  # noqa

    plt.xlabel("Theta={}".format(label))
    plt.xticks([-90, 0, 90])
    # plt.xticks(np.arange(-90, 90, 30))
    if idx == 0:
        plt.ylabel("Parameter loading")
        # plt.legend(loc="best")
    if idx > 0:
        ax.set_yticklabels([""])
        pass
    plt.ylim([-0.1, 1.1])
plt.show()
plt.close(f)
