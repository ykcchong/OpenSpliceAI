# import h5py
# # filename = "datafile_train_all.h5"
# # filename = "dataset_train_all.h5"
# filename = "dataset_test_0.h5"
# # filename = "datafile_test_0.h5"

# with h5py.File(filename, "r") as f:
#     # List all groups
#     print(("Keys: %s" % list(f.keys())))
#     a_group_key = list(f.keys())[3]
#     print(("a_group_key: ", a_group_key))

#     # Get the data
#     data = list(f[a_group_key])
#     # print(("data: ", data))
#     print(("data: ", len(data)))


import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

# hf = h5py.File('./dataset_train_all.h5', 'r')
hf = h5py.File('./dataset_test_0.h5', 'r')

hf.keys()

torch.from_numpy(hf['X0'][:]).shape
torch.from_numpy(hf['X0'][:])[0].float()
torch.from_numpy(hf['Y0'][:]).shape

x = torch.from_numpy(hf['X3'][:]).float()
y = torch.from_numpy(hf['Y3'][:])


print(f"x[0].shape: {x[0].shape}")

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(111)
ax.set_xlim([0, 15000])

ax.plot(x[len(x)-1].sum(axis=1))
print("x[0].sum(axis=1): ", len(x[len(x)-1].sum(axis=1)))
print("x[0].sum(axis=0): ", len(x[len(x)-1].sum(axis=0)))

plt.show()

print(f"(x[0].sum(axis=1) == 0).sum(): {(x[0].sum(axis=1) == 0).sum()}")

print(f"y[0].shape): {y[0].shape}")