#start
#import python modules
#pass
#import package modules
import pytest
import numpy as np
#import local modules
from library.utils import z_score_normalization
#end

def orig_z_score(x:np.ndarray) -> np.ndarray:
    mean = np.mean(x)
    std = np.std(x)
    z_scores = (x - mean) / std
    return z_scores

@pytest.mark.parametrize("args", [
    np.array([1, 2, 3,4,5,6,7,8,9]),
    np.array([70, 80, 90, 100, 110]),
    np.array([1, 2, 3]),
])
def test_z_score_normalization(args):
    expected = orig_z_score(args)
    result = z_score_normalization(args)
    np.testing.assert_allclose(result, expected)