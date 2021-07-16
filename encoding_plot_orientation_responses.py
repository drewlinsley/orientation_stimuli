import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from skimage import io
import sys


if len(sys.argv) > 2:
    file_name = sys.argv[1]
else:
    file_name = None
if len(sys.argv) > 3:
    no_surround = np.load(sys.argv[2])
    surround = np.load(sys.argv[3])
    surround_control = np.load(sys.argv[4])
else:
    no_surround = np.load('orientation_probe_no_surround_outputs_data.npy')
    surround_control = np.load('surround_control_outputs_data.npy')
    surround = np.load('orientation_probe_outputs_data.npy')

no_surround_images = 'orientation_probe_no_surround/test/imgs/1'
surround_images = 'orientation_probe/test/imgs/1'
surround_control_images = 'surround_control/test/imgs/1'
output_dir = "results_orientation_tuning"
os.makedirs(output_dir, exist_ok=True)

stride = np.floor(180 / no_surround.shape[0]).astype(int)
# thetas = {
#     0: -90,
#     30: -60,
#     60: -30,
#     90: 0,
#     120: 30,
#     150: 60
# }
thetas = {  # 0 theta is response to a blank scene
    1: -90,
    31: -60,
    61: -30,
    91: 0,
    121: 30,
    151: 60
}

ranges = np.arange(-90, 120, stride)  # [:no_surround.shape[0]]
# ranges = ranges[:no_surround.shape[1]]
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# No-surround
f = plt.figure(figsize=(13, 2))
plt.suptitle("Responses to no-surround gratings")
for idx, (theta, label) in enumerate(thetas.items()):
    theta = int(theta)
    ax = plt.subplot(2, len(thetas), idx + 1)

    it_image = io.imread(os.path.join(no_surround_images, "sample_{}.png".format(theta)))  # noqa
    plt.imshow(it_image[150:-150], cmap="Greys_r")
    plt.axis("off")

    ax = plt.subplot(2, len(thetas), idx + len(thetas) + 1)
    it_no_surround = no_surround[:, theta]
    it_no_surround = np.concatenate((it_no_surround, [it_no_surround[0]]))  # noqa
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
if file_name is not None:
    plt.savefig(os.path.join(output_dir, "{}_surround.pdf".format(file_name)))
else:
    plt.show()
plt.close(f)

# Surround
f = plt.figure(figsize=(13, 2))
plt.suptitle("Responses to no-surround gratings")
for idx, (theta, label) in enumerate(thetas.items()):
    theta = int(theta)
    ax = plt.subplot(2, len(thetas), idx + 1)

    it_image = io.imread(os.path.join(surround_images, "sample_{}.png".format(theta)))  # noqa
    plt.imshow(it_image[150:-150], cmap="Greys_r")
    plt.axis("off")

    ax = plt.subplot(2, len(thetas), idx + len(thetas) + 1)
    it_surround = surround[:, theta]
    it_surround = np.concatenate((it_surround, [it_surround[0]]))  # noqa
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
if file_name is not None:
    plt.savefig(os.path.join(output_dir, "{}_no_surround.pdf".format(file_name)))  # noqa
else:
    plt.show()
plt.close(f)

# Surround-control
f = plt.figure(figsize=(13, 2))
plt.suptitle("Responses to no-surround gratings")
for idx, (theta, label) in enumerate(thetas.items()):
    theta = int(theta)
    ax = plt.subplot(2, len(thetas), idx + 1)

    it_image = io.imread(os.path.join(surround_control_images, "sample_{}.png".format(theta)))  # noqa
    plt.imshow(it_image[150:-150], cmap="Greys_r")
    plt.axis("off")

    ax = plt.subplot(2, len(thetas), idx + len(thetas) + 1)
    it_surround = surround_control[:, theta]
    it_surround = np.concatenate((it_surround, [it_surround[0]]))  # noqa
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
if file_name is not None:
    plt.savefig(os.path.join(output_dir, "{}_modulation.pdf".format(file_name)))  # noqa
else:
    plt.show()
plt.close(f)
