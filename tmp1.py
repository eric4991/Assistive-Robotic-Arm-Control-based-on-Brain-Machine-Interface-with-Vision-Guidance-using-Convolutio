import numpy as np

a=np.zeros((5,5))
print(a)
a[2,3]=1
b=np.where(a==1)
print(b[0],b[1])
print(a[b])