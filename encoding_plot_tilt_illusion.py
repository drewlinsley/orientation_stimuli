import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from skimage import io
import sys


file_name = sys.argv[1]
surround = np.load(sys.argv[2])
surround_images = 'orientation_probe/test/imgs/1'
output_dir = "results_orientation_tilt_illusion"
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
    1: 0,
    11: 10,
    21: 20,
    31: 30,
    41: 40,
    51: 50,
    61: 60,
    71: 70,
    81: 80,
    91: 90,
}

ranges = np.arange(0, 90, stride)[:surround.shape[0]]
# ranges = ranges[:no_surround.shape[1]]
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# Surround
f = plt.figure(figsize=(13, 2))
plt.suptitle("Responses to no-surround gratings")
for idx, (theta, label) in enumerate(thetas.items()):
    theta = int(theta)
    ax = plt.subplot(2, len(thetas), idx + 1)
    import pdb;pdb.set_trace()
    it_image = io.imread(os.path.join(surround_images, "sample_{}.png".format(theta)))  # noqa
    plt.imshow(it_image, cmap="Greys_r")
    plt.axis("off")

    ax = plt.subplot(2, len(thetas), idx + len(thetas) + 1)
    it_surround = surround[:, theta]
    # plt.plot(ranges, it_surround, label="surround", color="Red", alpha=0.5, marker=".")  # noqa
    plt.plot(ranges, it_surround, label="no surround", color="Black", alpha=0.5, marker=".")  # noqa

    plt.xlabel("Theta={}".format(label))
    plt.xticks(ranges)
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
