

import numpy as np

def test(input):
    eegData=np.asarray(input)
    eegData=np.reshape(eegData,(2,3,2))
    print(eegData.shape)
    print(eegData)
    return eegData