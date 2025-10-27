#start
#import python modules
#pass
#import package modules
import numpy as np
#import local modules
#pass
#end

# class ZScore:
#     def __call__(self,x:np.ndarray) -> np.ndarray:
#         mean = np.mean(x)
#         std_dev = np.std(x)
#         z_scores = (x - mean) / std_dev
#         return z_scores

def z_score_normalization(x:np.ndarray) -> np.ndarray:
    mean = np.mean(x)
    std_dev = np.std(x)
    z_scores = (x - mean) / std_dev
    return z_scores