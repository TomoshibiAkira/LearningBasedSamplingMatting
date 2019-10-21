import torch
import numpy as np

in_a = np.arange(16)
in_a = np.resize(in_a, (4,4))

in_tensor = np.expand_dims(in_a, 0)
in_tensor = np.expand_dims(in_tensor, 0)

print("input: ")
print(in_tensor)

in_t = torch.from_numpy(in_tensor)
all_patches = in_t.unfold(2, 2, 1).unfold(3, 2, 1)

print("all patches")
print(all_patches[0][0][0][1])
print(all_patches.shape)