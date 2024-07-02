"""Contains classes for defining failure."""

from abc import ABC, abstractmethod
from itertools import combinations
import random
from typing import Union, List, Dict, Tuple, Callable
import numpy as np
import sympy as sp

class FailureSet(ABC):
    @abstractmethod
    def contains(self, x: np.ndarray) -> bool:
        """Check if a given point is in the failure set.
        
        Args:
            x (np.ndarray): A numpy array representing the point.
        """
        ...

    def depth(self, x: np.ndarray) -> float:
        """Calculate the depth of a point within the failure set.
        
        Args:
            x (np.ndarray): A numpy array representing the point.
        """
        if self.contains(x):
            return abs(self.dist(x))
        else:
            return 0

    @abstractmethod
    def dist(self, x: np.ndarray) -> float:
        """Calculate the distance of a point to the boundary of the failure set.
        
        Args:
            x (np.ndarray): A numpy array representing the point.
        """
        ...

class ClosedHalfSpaceFailureSet(FailureSet):
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
    
    def contains(self, x: Dict[str, float]) -> bool:
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


class BooleanFormulaFailureSet(FailureSet):
    def __init__(self, failure_formual: sp.Basic):
        """ Represents a failure set defined by a Boolean formula.
        
        Args:
            failure_formual (sp.FunctionClass): A sympy function representing the failure set.
        
        Example:
            >>> from sympy import symbols
            >>> x, y = symbols('x y')
            >>> f = x & y
            >>> bfs = BooleanFormulaFailureSet(f)
        """
        if not isinstance(failure_formual, sp.Basic):
            raise ValueError("The failure_formual must be a sympy object.")
        self.failure_formual = failure_formual
        self.vars_order = list(str(x) for x in failure_formual.free_symbols)
    
    @staticmethod
    def get_random(vars: list[str], seed: int = 42):
        syms = [sp.symbols(v) for v in vars]
        bf = syms[0]
        rnd = random.Random(seed)
        for i in range(1,len(vars)):
            op = rnd.choice([sp.And, sp.Or])
            do_not = rnd.choice([True, False])
            if do_not:
                bf = op(bf, sp.Not(syms[i]))
            else:
                bf = op(bf, syms[i])

        return BooleanFormulaFailureSet(bf)
    
    def contains(self, x: Dict[str, float]) -> bool:
        """Check if a given point is in the failure set.
        
        Args:
            x (Dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        x = {k: bool(v) for k, v in x.items()}
        return bool(self.failure_formual.subs(x))

    def dist(self, x: Dict[str, float]) -> float:
        """Calculate the Hamming distance of a point to the boundary of the failure set.
        
        The result is always non-negative.

        Args:
            x (Dict[str, float]): A dictionary representing the point with keys as the variable names.
        """
        initial_result = self.contains(x)
        n = len(self.vars_order)
        for changes in range(1, n+1):
            for combo in combinations(self.vars_order, changes):
                toggled_x = x.copy()
                for var in combo:
                    toggled_x[var] = not toggled_x[var]
                if self.contains(toggled_x) != initial_result:
                    return changes
        return n  # In the worst case, all variables need to be toggled
    
    def __str__(self) -> str:
        return f'BooleanFormulaFailureSet({self.failure_formual=})'

def test_closed_half_space_failure_set():
    hs = ClosedHalfSpaceFailureSet({'A': (0, 'ge'), 'B': (0, 'le'), 'C': (0, 'ge')})  # Define a half-space in R^3
    assert hs.contains({'A': 1, 'B': -1, 'C': 1})  # Should be True if x >= 0, y <= 0, z >= 0
    assert not hs.contains({'A': -1, 'B': 1, 'C': -1}) # Should print False
    assert hs.dist({'A': 1, 'B': 1, 'C': 1}) == 1
    assert hs.dist({'A': -1, 'B': -1, 'C': -2}) == 1

    hs2 = ClosedHalfSpaceFailureSet({'C': (200, 'ge')})  # Define a half-space in R^3
    assert hs2.dist({'A': -1, 'B': -1, 'C': 0}) == 200
    assert hs2.dist({'A': 100, 'B': -1, 'C': 10}) == 190
    
    
def test_boolean_formula_failure_set():
    x, y = sp.symbols('x y')
    f = x & y
    bfs = BooleanFormulaFailureSet(f)
    assert bfs.contains({'x': True, 'y': True})
    assert not bfs.contains({'x': True, 'y': False})
    assert 1 == bfs.dist({'x': True, 'y': False})
    assert 1 == bfs.dist({'x': False, 'y': True})
    assert 2 == bfs.dist({'x': False, 'y': False})

    x, y, z = sp.symbols('x y z')
    f = x & y & z
    bfs = BooleanFormulaFailureSet(f)
    assert bfs.contains({'x': True, 'y': True, 'z': True})
    assert not bfs.contains({'x': True, 'y': True, 'z': False})
    assert 1 == bfs.dist({'x': True, 'y': True, 'z': False})
    assert 2 == bfs.dist({'x': False, 'y': False, 'z': True})
    assert 3 == bfs.dist({'x': False, 'y': False, 'z': False})

if __name__ == "__main__":
    test_closed_half_space_failure_set()
    test_boolean_formula_failure_set()
    print("All tests passed!")