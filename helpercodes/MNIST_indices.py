import numpy as np

data = np.load('indices/mnist_test_aug_indices.npy')
# a= 10733
data = np.sort(data)
data[::-1].sort()

# if a in data:
#     print ("yes")
print (f"data len {data.shape}  {type(data)}")

arr = np.arange(10000)

for i in data:
    arr = np.delete(arr ,i)

# print (f"data len {arr.shape}  {type(arr)}")
np.save('indices/mnist_test_non_aug_indices.npy', arr)