import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from skimage import io


no_surround = np.load('orientation_probe_no_surround_outputs_data.npy')
no_surround_images = 'orientation_probe_no_surround/test/imgs/1'
surround = np.load('orientation_probe_outputs_data.npy')
surround_images = 'orientation_probe/test/imgs/1'

stride = np.floor(180 / no_surround.shape[-1]).astype(int)
# thetas = {
#     0 // stride: -90,
#     30 // stride: -60,
#     60 // stride: -30,
#     90 // stride: 0,
#     120 // stride: 30,
#     150 // stride: 60
# }
thetas = {
    0: -90,
    30: -60,
    60: -30,
    90: 0,
    120: 30,
    150: 60
}

ranges = np.arange(-90, 90, stride)
ranges = ranges[:no_surround.shape[1]]
print(surround.shape)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# No-surround
f = plt.figure(figsize=(13, 2))
plt.suptitle("Responses to no-surround gratings")
for idx, (theta, label) in enumerate(thetas.items()):
    theta = int(theta)
    ax = plt.subplot(2, len(thetas), idx + 1)

    it_image = io.imread(os.path.join(no_surround_images, "sample_{}.png".format(theta)))  # noqa
    plt.imshow(it_image, cmap="Greys_r")
    plt.axis("off")

    ax = plt.subplot(2, len(thetas), idx + len(thetas) + 1)
    it_no_surround = no_surround[theta]
    # plt.plot(ranges, it_surround, label="surround", color="Red", alpha=0.5, marker=".")  # noqa
    plt.plot(ranges, it_no_surround, label="no surround", color="Black", alpha=0.5, marker=".")  # noqa

    # Draw verticle lines
    # plt.axvline(x=ranges[np.argmax(it_surround)], color="Red", linestyle="--", alpha=0.5)  # noqa
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

# Surround
# No-surround
f = plt.figure(figsize=(13, 2))
plt.suptitle("Responses to no-surround gratings")
for idx, (theta, label) in enumerate(thetas.items()):
    theta = int(theta)
    ax = plt.subplot(2, len(thetas), idx + 1)

    it_image = io.imread(os.path.join(surround_images, "sample_{}.png".format(theta)))  # noqa
    plt.imshow(it_image, cmap="Greys_r")
    plt.axis("off")

    ax = plt.subplot(2, len(thetas), idx + len(thetas) + 1)
    it_surround = surround[theta]
    # plt.plot(ranges, it_surround, label="surround", color="Red", alpha=0.5, marker=".")  # noqa
    plt.plot(ranges, it_surround, label="no surround", color="Black", alpha=0.5, marker=".")  # noqa

    # Draw verticle lines
    # plt.axvline(x=ranges[np.argmax(it_surround)], color="Red", linestyle="--", alpha=0.5)  # noqa
    plt.axvline(x=ranges[np.argmax(it_surround)], color="Black", linestyle="--", alpha=0.5)  # noqa

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
