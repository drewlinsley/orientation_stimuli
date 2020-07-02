import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


surround = np.load('contrast_modulated_outputs_data.npy')
no_surround = np.load('contrast_modulated_no_surround_outputs_data.npy')

stride = np.floor(180 / surround.shape[-1]).astype(int)
stride = np.maximum(stride, 1)
# thetas = {0: -90, 30//5: -60, 60//5: -30, 90//5: 0, 120//5: 30, 150//5: 60}
thetas = {
    0 // stride: -90,
    30 // stride: -60,
    60 // stride: -30,
    90 // stride: 0,
    120 // stride: 30,
    150 // stride: 60
}
metas = np.load("contrast_modulated_no_surround_outputs/1.npy", allow_pickle=True, encoding="latin1")  # noqa
metas = metas.reshape(-1, 14)
mask = (np.asarray(metas[:, 4]).astype(int) == -45)
metas = metas[mask]
no_surround = no_surround[mask]
surround = surround[mask]
contrasts = (metas[:, -2:]).astype(float)

# no_surround = no_surround.reshape(5, 5, 180)  # [..., 0]
# surround = surround.reshape(5, 5, 180)  # [..., 0]

# Roll so that they start at -45
no_surround = np.roll(no_surround, 45, axis=-1)
surround = np.roll(surround, 45, axis=-1)

ranges = np.arange(-45, 135, stride)
ranges = ranges[:surround.shape[-1]]

idx3a = np.asarray([0, 5, 10, 20, 2, 7, 12, 17])

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
f = plt.figure(figsize=(7, 4))
max_surround, max_surround_idx = [], []
max_no_surround, max_no_surround_idx = [], []
img_data = np.load("images_contrast_modulated_no_surround_outputs_data.npy")
plt.suptitle("Busse and Wade Fig 3a")
count = 1
for c in range(4):
    for r in range(2):
        ax = plt.subplot(4, 4, count)
        it_no_surround = no_surround[idx3a[count - 1]]
        ax.plot(ranges, it_no_surround, label="no surround", color="Black", alpha=0.5, marker=".")  # noqa
        plt.title(str(contrasts[count - 1]))
        if r + c == 0:
            plt.ylabel("Parameter loading")
            # plt.legend(loc="best")
        if r + c > 0:
            # ax.set_yticklabels([""])
            pass
        if r == 4:
            plt.xticks([-45, 0, 45, 90, 135])
        else:
            ax.set_xticklabels([""])
        plt.ylim([-0, 1])

        # Now load the image
        ax = plt.subplot(4, 4, count + 8)
        ax.imshow(img_data[idx3a[count - 8], ..., 0], cmap="Greys_r")
        ax.axis("off")
        count += 1
plt.show()
plt.close(f)



sns.set_style("darkgrid", {"axes.facecolor": ".9"})
f = plt.figure(figsize=(7, 7))
max_surround, max_surround_idx = [], []
max_no_surround, max_no_surround_idx = [], []
plt.suptitle("Trott and Born Fig 3C")
count = 1
for c in range(5):
    for r in range(5):
        ax = plt.subplot(5, 5, count)
        it_no_surround = no_surround[count - 1]
        ax.plot(ranges, it_no_surround, label="no surround", color="Black", alpha=0.5, marker=".")  # noqa
        plt.title(str(contrasts[count - 1]))
        count += 1
        if r + c == 0:
            plt.ylabel("Parameter loading")
            # plt.legend(loc="best")
        if r + c > 0:
            # ax.set_yticklabels([""])
            pass
        if r == 4:
            plt.xticks([-45, 0, 45, 90, 135])
        else:
            ax.set_xticklabels([""])
        plt.ylim([-0, 1])
plt.show()
plt.close(f)
