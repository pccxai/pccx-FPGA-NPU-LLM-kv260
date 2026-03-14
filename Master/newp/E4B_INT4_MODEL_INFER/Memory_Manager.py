import numpy as np

def allocate_KVcache(layers, token, dimension):
    # layers x size x dimension x type = arrsize
    A = np.zeros((layers, token, dimension),dtype=np.float16)
    return A

if __name__ == "__main__":
    A = allocate_KVcache(35,2048,512)
    print(A.shape)