import numpy as np
# yeild 2**k x 2**k Hadamard matrix(for 4096 k=12)
def generateHadamard (k):
    h = np.array([[1,1],[1,-1]])
    H = h
    for i in range(k)[1:]:
        H = np.kron(H,h)
    return H
# yeild projecton matrix HD
def HD(k):
    D =np.zeros(shape = (2**k,2**k))
    index = range(2**k)
    D[index,index] = np.random.randint(2)*2-1
    H = generateHadamard(k) 
    Mat = np.dot(H,D)
    return Mat

 #usage: S = HD(12)   12 for 4096 (2**12==4096)