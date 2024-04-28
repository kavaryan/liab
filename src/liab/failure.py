"""Contains classes for defining failure."""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Tuple, Callable
import numpy as np

class FailureSet(ABC):
    @abstractmethod
    def conatins(self, x: np.ndarray) -> bool:
        """Check if a given point is in the failure set.
        
        Args:
            x (np.ndarray): A numpy array representing the point.
        """
        pass

    def depth(self, x: np.ndarray) -> float:
        """Calculate the depth of a point within the failure set.
        
        Args:
            x (np.ndarray): A numpy array representing the point.
        """
        if self.conatins(x):
            return abs(self.dist(x))
        else:
            return 0

    @abstractmethod
    def dist(self, x: np.ndarray) -> float:
        """Calculate the distance of a point to the boundary of the failure set.
        
        Args:
            x (np.ndarray): A numpy array representing the point.
        """
        pass

class ClosedHalfSpace(FailureSet):
    def __init__(self, bounadry_dict: Dict[str, Tuple[float, str]]):
        """ Represents a closed half-space in n-dimensional space in the form 
            x_i >= a_i or x_i <= b_i for i = 1, 2, ..., n.
        
        Args:
            boundary_dict (Dict[str, float]): A dictionary representing the boundaries:
                key: the name of the variable, value: the boundary value and type
                    boundary value: a float representing the value of the boundary,
                    boundary type: a string ('ge' for >=, 'le' for <=)
                        defining the type of inequality for each boundary.
        """
        self.boundary_dict = bounadry_dict
    
    def conatins(self, x: Dict[str, float]) -> bool:
        """Check if a given point is in the failure set.
        
        Args:
            x (Dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        for k, v in self.boundary_dict.items():
            if v[1] == 'ge':
                if x[k] < v[0]:
                    return False
            elif v[1] == 'le':
                if x[k] > v[0]:
                    return False
        return True

    def dist(self, x: Dict[str, float]) -> float:
        """Calculate the distance of a point to the boundary of the failure set.
        
        The result is always non-negative.

        Args:
            x (Dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        return min([abs(x[k] - v[0]) for k, v in self.boundary_dict.items()])

def test_closed_half_space():
    hs = ClosedHalfSpace({'A': (0, 'ge'), 'B': (0, 'le'), 'C': (0, 'ge')})  # Define a half-space in R^3
    assert hs.conatins({'A': 1, 'B': -1, 'C': 1})  # Should be True if x >= 0, y <= 0, z >= 0
    assert not hs.conatins({'A': -1, 'B': 1, 'C': -1}) # Should print False
    assert hs.dist({'A': 1, 'B': 1, 'C': 1}) == 1
    assert hs.dist({'A': -1, 'B': -1, 'C': -2}) == 1

    hs2 = ClosedHalfSpace({'C': (200, 'ge')})  # Define a half-space in R^3
    assert hs2.dist({'A': -1, 'B': -1, 'C': 0}) == 200
    assert hs2.dist({'A': 100, 'B': -1, 'C': 10}) == 190
    
    print("all tests passed!")

if __name__ == "__main__":
    test_closed_half_space()