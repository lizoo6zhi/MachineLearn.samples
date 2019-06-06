import numpy as np

li = [4,9,16,25]
m = np.mat(np.array(li).reshape(2,2))
print('m:',m)
de = np.linalg.det(m)
print('行列式：',de)
#特征值和特征向量
f_v,f_a = np.linalg.eig(m)
print('特征值:',f_v)
print('特征向量:',f_a)
bb = np.sqrt(np.diag(f_v))
print(bb)
aa = f_a.dot(bb).dot(np.linalg.inv(f_a))
print(aa)
