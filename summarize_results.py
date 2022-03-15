import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from glob import glob
import pandas as pd
from collections import OrderedDict
from scipy.signal import savgol_filter


w = 3  # 9
p = 1
exclude = 0  # 1

tb_fig4 = glob(os.path.join("results_bwc_fig4a/*diff_scores*"))
tb_fig2 = glob(os.path.join("results_tb_fig2_diff/*full_scores*"))
bwc_fig4 = glob(os.path.join("results_tb_fig4b_full/*full_scores*"))
perfs = glob("/media/data_cifs/projects/prj_neural_circuits/py-bsds500/results/ckpt-*")

# Exclude where model is not in the name
tb_fig4 = [x for x in tb_fig4 if "model" in x]
tb_fig2 = [x for x in tb_fig2 if "model" in x]
bwc_fig4 = [x for x in bwc_fig4 if "model" in x]

tb_fig4 = sorted(tb_fig4, key=os.path.getmtime)[::-1]
tb_fig2 = sorted(tb_fig2, key=os.path.getmtime)[::-1]
bwc_fig4 = sorted(bwc_fig4, key=os.path.getmtime)[::-1]
pidx = [int(x.split("ckpt-")[-1].split("-")[0]) for x in perfs]
pidx = np.argsort(pidx)
perfs = np.asarray(perfs)[pidx]

tb_fig4 = tb_fig4[exclude:]
tb_fig2 = tb_fig2[exclude:]
bwc_fig4 = bwc_fig4[exclude:]
perfs = perfs[exclude:]
# tb_fig4 = np.concatenate((tb_fig4[:316], tb_fig4[364:]))
# tb_fig2 = np.concatenate((tb_fig2[:316], tb_fig2[364:]))
# bwc_fig4 = np.concatenate((bwc_fig4[:316], bwc_fig4[364:]))
# perfs = np.concatenate((perfs[:316], perfs[364:]))

print(len(tb_fig4))
print(len(tb_fig2))
print(len(bwc_fig4))
print(len(perfs))

tb_fig4_data = OrderedDict()
# diff_tb_fig4_data = OrderedDict()
tb_fig2_data = OrderedDict()
bwc_fig4_data = OrderedDict()
perf_data = OrderedDict()
for f in tb_fig4:
    d = float(np.load(f)[0])
    k = f.split("model_")[-1].split("_scores")[0]
    n = int(f.split("model_")[-1].split("_scores")[0].split("_")[0])
    # tb_fig4_data[k] = d
    tb_fig4_data[n] = d
    # diff_tb_fig4_data[n] = float(np.load(f)[1])
   
for f in tb_fig2:
    d = float(np.load(f)[0])
    k = f.split("model_")[-1].split("_scores")[0]
    n = int(f.split("model_")[-1].split("_scores")[0].split("_")[0])
    # tb_fig2_data[k] = d
    tb_fig2_data[n] = d

for f in bwc_fig4:
    d = float(np.load(f)[0])
    k = f.split("model_")[-1].split("_scores")[0]
    n = int(f.split("model_")[-1].split("_scores")[0].split("_")[0])
    # bwc_fig4_data[k] = d 
    bwc_fig4_data[n] = d

for f in perfs:
    d = np.load(f)
    data = d['best_f1']
    del d.f
    d.close()
    n = int(f.split(os.path.sep)[-1].split(".")[0].split("-")[1])
    perf_data[n] = data
perf_keys = perf_data.keys()
perf_data = np.asarray([x for x in perf_data.values()])  #  * 100
# perf_data = savgol_filter(perf_data, w, p)

# Get the best perf checkpoint
# arg_ckpt = np.argmax(perf_data)
# arg_ckpt = 622
# arg_ckpt = np.argmax([x for x in diff_tb_fig4_data.values()])
# INSILICO_bsds_perfs_v2_val/gammanet_bsds/ckpt-311000

# arg_perf = np.load(perfs[arg_ckpt])['best_f1']
# arg_perf = np.load([x for x in perfs if "311000" in x][0])
arg_ckpt = 343500
arg_perf = np.load([x for x in perfs if "343500" in x][0])["best_f1"]

import pdb;pdb.set_trace()
arg_exp2 = tb_fig2_data[arg_ckpt]  # [x for x in tb_fig2_data.values()][arg_ckpt]
arg_exp4 = tb_fig4_data[arg_ckpt]  # [x for x in tb_fig4_data.values()][arg_ckpt]
arg_bwc = bwc_fig4_data[arg_ckpt]  # [x for x in bwc_fig4_data.values()][arg_ckpt]
# print("Best ckpt: {}, perf {} exp2 {} exp4 {} bwc {}".format([x for x in perf_keys][::-1][arg_ckpt], arg_perf, arg_exp2, arg_exp4, arg_bwc))
print("Best ckpt: {}, perf {} exp2 {} exp4 {} bwc {}".format(343500, arg_perf, arg_exp2, arg_exp4, arg_bwc))

plt.subplot(242)
z = [x for x in tb_fig4_data.values()][::-1]
yhat = savgol_filter(z, w, p)
plt.plot(z)  # yhat)
plt.ylim([0, 1])
plt.title("TB Fig 4")
plt.subplot(241)
plt.ylabel(r"$R^2$ model vs primate")
z = [x for x in tb_fig2_data.values()][::-1]
yhat = savgol_filter(z, w, p)
plt.plot(z)  # yhat)
plt.ylim([0, 1])
plt.title("TB Fig 2")
plt.subplot(243)
z = [x for x in bwc_fig4_data.values()][::-1]
yhat = savgol_filter(z, w, p)
plt.plot(z)  # yhat)
plt.ylim([0, 1])
plt.title("BWC Fig 4")
plt.subplot(244)
# z = [x for x in perf_data.values()]
z = perf_data
# yhat = savgol_filter(z, w, p)
plt.plot(z)  # yhat)
plt.ylim([0, 1])
plt.title("ODS")

plt.subplot(245)
z = [x for x in tb_fig4_data.values()][::-1]
# y = [x for x in perf_data.values()]
y = perf_data
plt.plot(z, y, ".")
plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.title("TB Fig 4")
plt.subplot(246)
z = [x for x in tb_fig2_data.values()][::-1]
# y = [x for x in perf_data.values()]
y = perf_data
plt.plot(z, y, ".")
plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.title("TB Fig 2")
plt.subplot(247)
z = [x for x in bwc_fig4_data.values()][::-1]
# y = [x for x in perf_data.values()]
y = perf_data
plt.plot(z, y, ".")
plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.title("BWC Fig 4")
plt.subplot(248)
# z = [x for x in perf_data.values()]
z = perf_data
# yhat = savgol_filter(z, w, p)
plt.plot(np.arange(len(z)), np.exp(z), ".")
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.title("ODS")

plt.show()

