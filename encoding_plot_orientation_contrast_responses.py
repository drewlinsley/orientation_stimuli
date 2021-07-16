import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from skimage import io
import sys


file_name = sys.argv[1]
surround = np.load(file_name)
surround_images = 'orientation_probe/test/imgs/1'
output_dir = "results_orientation_tuning_contrast"
os.makedirs(output_dir, exist_ok=True)

stride = np.floor(180 / surround.shape[0]).astype(int)
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

# Surround 06
f = plt.figure(figsize=(13, 2))
it_file_name = file_name
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
    plt.savefig(os.path.join(output_dir, "{}_contrast.pdf".format(it_file_name.split(os.path.sep)[-1])))  # noqa
else:
    plt.show()
plt.close(f)

# Surround 12
it_file_name = file_name.replace("06", "12")
surround = np.load(it_file_name)
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
    plt.savefig(os.path.join(output_dir, "{}_contrast.pdf".format(it_file_name.split(os.path.sep)[-1])))  # noqa
else:
    plt.show()
plt.close(f)

# Surround 25
it_file_name = file_name.replace("06", "25")
surround = np.load(it_file_name)
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
    plt.savefig(os.path.join(output_dir, "{}_contrast.pdf".format(it_file_name.split(os.path.sep)[-1])))  # noqa
else:
    plt.show()
plt.close(f)

# Surround 50
it_file_name = file_name.replace("06", "50")
surround = np.load(it_file_name)
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
    plt.savefig(os.path.join(output_dir, "{}_contrast.pdf".format(it_file_name.split(os.path.sep)[-1])))  # noqa
else:
    plt.show()
plt.close(f)

# Surround 75
it_file_name = file_name.replace("06", "75")
surround = np.load(it_file_name)
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
    plt.savefig(os.path.join(output_dir, "{}_contrast.pdf".format(it_file_name.split(os.path.sep)[-1])))  # noqa
else:
    plt.show()
plt.close(f)
