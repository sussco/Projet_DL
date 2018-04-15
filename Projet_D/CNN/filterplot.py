import numpy as np
import matplotlib.pyplot as plt


filter1 = np.array([[[-0.02912225],
  [-0.03315046],
  [ 0.00967736]],

 [[ 0.00107062],
  [-0.00715093],
  [-0.00708249]],

 [[ 0.01282419],
  [-0.02135836],
  [-0.00088052]]])

filter2 = np.array([[[ 0.00301456],
  [ 0.00346279],
  [-0.01651029]],

 [[ 0.00346948],
  [-0.0111207 ],
  [ 0.00516575]],

 [[-0.00063214],
  [-0.00877737],
  [-0.00739115]]])

filter3 = np.array([[[-0.0221143 ],
  [-0.00050929],
  [-0.02537607]],

 [[-0.00438778],
  [-0.02778603],
  [-0.02946103]],

 [[-0.01843826],
  [-0.02749033],
  [ 0.00745133]]]
)
filter1 = np.reshape(filter1, (3,3))
filter2 = np.reshape(filter2, (3,3))
filter3 = np.reshape(filter3, (3,3))

plt.matshow(filter1,cmap=plt.cm.gray)
plt.matshow(filter2,cmap=plt.cm.gray)
plt.matshow(filter3,cmap=plt.cm.gray)



plt.show()
