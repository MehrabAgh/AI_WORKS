import numpy as np

arr = np.array([1,2,3,4,5,6,7,8,9] , type(float()))

# numpy attribute

print("dim : " , arr.ndim )

print("shape : " , arr.shape)

print("length : " , arr.size )

print("type items : " , arr.dtype)

print("size items (per byte) : " , arr.itemsize )

print("buffer of data  : " , arr.data)

print("sum items  : " , arr.sum() )

print("min items  : " , arr.min() )

print("max items  : " , arr.max() , "\n")

# numpy function

print("type func : ", type(arr) , "\n")

print("zeros item for array func : ", np.zeros((4,2)) , "\n")

print("ones item for array func : ", np.ones((4,2)), "\n")

print("empty func : ", np.empty([4,4]) , "\n")

print("random arange func : ", np.arange(0,5,1), "\n")

print("random linspace func : ", np.linspace(0, 20 , 3) , "\n")

print("random logspace func : ", np.logspace(0 , 20 , 5), "\n" )

x =  np.logspace(0 , 20 , 5)
print("sin logspace func : ", np.sin(x) , "\n")

print("reshape func : ", arr.reshape(1,9), "\n")

print("random number func : ", np.random.random((0, 20)) , "\n")

print("exponential item array func : ", np.exp(arr) , "\n")

print("radical item array func : ", np.sqrt(arr), "\n")


