import numpy as np
from matplotlib import pyplot as plt


x = it_train_dict['fgru']
gt = it_train_dict['train_labels'].ravel()[-6:]

##
# Load tuning curve transform
moments_file = "../undo_bias/neural_models/linear_moments/tb_feature_matrix.npz"
model_file = "../undo_bias/neural_models/linear_models/tb_model.joblib.npy"
moments = np.load(moments_file)
means = moments["means"]
stds = moments["stds"]
clf = np.load(model_file).astype(np.float32)

# Transform activity to outputs
bs, h, w, _ = x.shape
hh, hw = h // 2, w // 2
sel_units = np.reshape(x[:, hh - 2: hh + 2, hw - 2: hw + 2, :], [bs, -1])

# Normalize activities
# sel_units = (sel_units - means) / stds

# Map responses
inv_clf = np.linalg.inv(clf.T.dot(clf))
inv_matmul = inv_clf.dot(clf.T)
activity = inv_matmul.dot(sel_units.T)

# Invert the responses to activities
inv_inv = np.linalg.pinv(clf.dot(clf.T))
tc_inv = activity.T.dot(clf.T).dot(clf).dot(clf.T).dot(inv_inv)
tc_inv = tc_inv.reshape(-1)

# Unnormalize activities
ntc_inv = tc_inv * stds + means
print(gt)
print(activity.squeeze())

print(np.argmax(gt))
print(np.argmax(activity.squeeze()))

from matplotlib import pyplot as plt
f = plt.figure()
plt.plot(activity.squeeze(), label="exp")
plt.plot(gt, label="gt")
plt.legend()
plt.show()



from matplotlib import pyplot as plt
f = plt.figure()
plt.plot(it_train_dict["train_logits"].squeeze(), label="exp")
plt.plot(it_train_dict["train_labels"].ravel()[-6:], label="gt")
plt.legend()
plt.show()

