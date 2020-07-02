import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats, interpolate
import pandas as pd




show_truncated = True
no_surround = np.load('plaid_no_surround_outputs_data.npy')
surround = np.load('plaid_surround_outputs_data.npy')
# import ipdb;ipdb.set_trace()
# ds_surround = interpolate.interp1d(np.arange(surround.shape[-1]), surround, "linear")
# ds_no_surround = interpolate.interp1d(np.arange(no_surround.shape[-1]), no_surround, "linear")
# surround = ds_surround(np.arange(0, len(surround), 6))
# no_surround = ds_no_surround(np.arange(0, len(no_surround), 6))

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

# thetas = {
#     0 // stride: -90,
#     1 // stride: -60,
#     2 // stride: -30,
#     3 // stride: 0,
#     4 // stride: 30,
#     5 // stride: 60
# }


ranges = np.arange(-90, 90, stride)
ranges = ranges[:surround.shape[1]]

print(no_surround.shape)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
f = plt.figure(figsize=(13, 2))
max_surround, max_surround_idx = [], []
max_no_surround, max_no_surround_idx = [], []
plt.suptitle("Trott and Born Fig 3C")
for idx, (theta, label) in enumerate(thetas.items()):
    theta = int(theta)
    ax = plt.subplot(1, len(thetas), idx + 1)

    it_surround = surround[theta]
    it_no_surround = no_surround[theta]
    plt.plot(ranges, it_surround, label="surround", color="Red", alpha=0.5, marker=".")  # noqa
    plt.plot(ranges, it_no_surround, label="no surround", color="Black", alpha=0.5, marker=".")  # noqa

    max_surround.append(it_surround.max())
    max_surround_idx.append(np.argmax(it_surround))
    max_no_surround.append(it_no_surround.max())
    max_no_surround_idx.append(np.argmax(it_no_surround))

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

# Prepare for stats
max_surround_idx = np.abs(np.asarray(max_surround_idx) - 90)
max_no_surround_idx = np.abs(np.asarray(max_no_surround_idx) - 90)
max_surround = np.asarray(max_surround)
max_no_surround = np.asarray(max_no_surround)

# Main effect of no_surround v surround
me_diff = (max_no_surround - max_surround)
me_t, me_p = stats.ttest_1samp(me_diff, 0)

# Surround suppression of the non-surround grating
ss_diff = np.abs(max_surround_idx - max_no_surround_idx)
ss_t, ss_p = stats.ttest_1samp(max_surround_idx - max_no_surround_idx, 0)

# Plot the two bars
both = np.concatenate(
    (
        np.stack((me_diff, np.zeros_like(me_diff)), -1),
        np.stack((ss_diff, np.ones_like(ss_diff)), -1)), 0)
# plt.figure()
# df = pd.DataFrame(both, columns=["vals", "idx"])
# sns.barplot(data=df[df.idx == 0], x="idx", y="vals", color="Red", saturation=0.5)
# plt.title('Main effect')
# plt.ylim([-1, 1])
# plt.show()
# plt.close(f)

# plt.figure()
# sns.barplot(data=df[df.idx == 1], x="idx", y="vals", color="Red", saturation=0.5)
# plt.title('SS effect')
# plt.ylim([-30, 30])
# plt.show()
# plt.close(f)
if show_truncated:
    truncated_ranges = np.linspace(-90, 60, 6)
    truncated_idx = np.asarray([*thetas]).astype(int)
    # Downsample Dimensions to 7
    down_surround = surround[truncated_idx]
    down_no_surround = no_surround[truncated_idx]
    f = plt.figure(figsize=(13, 4))
    for idx, (theta, label) in enumerate(thetas.items()):
        theta = int(theta)
        ax = plt.subplot(1, len(thetas), idx + 1)

        plt.plot(truncated_ranges, (surround[theta, truncated_idx]), label="surround", color="Red", alpha=0.5, marker=".")  # noqa
        plt.plot(truncated_ranges, (no_surround[theta, truncated_idx]), label="no surround", color="Black", alpha=0.5, marker=".")  # noqa

        plt.xlabel("Theta={}".format(label))
        plt.xticks([-90, 0, 90])
        # plt.xticks(np.arange(-90, 90, 30))
        if idx == 0:
            plt.legend(loc="best")
        if idx > 0:
            # ax.set_yticklabels([""])
            pass
        # plt.ylim([-0.1, 1.1])
    plt.show()
