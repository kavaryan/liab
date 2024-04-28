"""Define classes for defining failure."""

from typing import Union, List, Dict, Tuple, Callable
import functools
import operator
from matplotlib import pyplot as plt

import numpy as np
from numpy.linalg import norm
import sympy.stats
from sympy import Max
import networkx as nx

GSym = Union[sympy.Symbol, sympy.stats.rv.RandomSymbol, str]

def flatten(a):
    return functools.reduce(operator.iconcat, a, [])

class ComponentOrEquation:
    def __init__(self, I: List[str], O: str, f: str):
        """Represent a component in a system or an equation in an SCM.
        
        Args:
            I (List[str]): Input variables.
            O (str): Output variable.
            f (str): Function of the output variable.
        """
        self.I = I
        self.O = O
        self.f = f

    def __repr__(self) -> str:
        return f"ComponentOrEquation(I={self.I}, O={self.O}, f={self.f})"

class System:
    def __init__(self, cs: List[ComponentOrEquation]):
        """Represnt a multi-compoent system.
        
        Args:
            cs (List[Component]): List of components.
        """
        assert len(set([c.O for c in cs])) == len(cs) # No duplicate output variables
        self.cs = cs

    def induced_scm(self, state_order: List[str] = None) -> "SCM":
        """Get the induced SCM of a system.
        
        Args:
            state_order (List[str], optional): Order of the states passed to SCM constructor.
        """
        V = set([c.O for c in self.cs])
        all = set(flatten([c.I for c in self.cs]))
        U = set(all) - set(V)
        return SCM(U, V, self.cs, state_order)
    
    def is_compatible(self, S: "System") -> bool:
        """Check if the system is compatible with a specification.

        The compoenents are compared in the order that they appear in their cs property.
        
        Args:
            S (System): The specification system.
        """
        if len(self.cs) != len(S.cs):
            return False
        return all(c.I == d.I and c.O == d.O for c, d in zip(self.cs, S.cs))
    
    def get_replacement(self, rep_dict: Dict[str, ComponentOrEquation]) -> "System":
        assert isinstance(rep_dict, dict)
        cs = []
        for c in self.cs:
            if c.O in rep_dict:
                cs.append(rep_dict[c.O])
            else:
                cs.append(c)
        return System(cs)
    
    def __repr__(self) -> str:
        return f"System(cs={self.cs})"

class SCM:
    def __init__(self, U: List[str], V: List[str], 
                 cs: List[ComponentOrEquation], state_order: List[str] = None):
        """Represnet a structural causal model (SCM).
        
        Args:
            U (List[str]): Exogenous variables.
            V (List[str]): Indogenous variables.
            cs (List[Component]): List of components.
            state_order (List[str], optional): Order of the states passed to SCM constructor,
                useful when constructing an implementation from an specification.
        """
        self.U = U
        self.V = V
        self.cs_dict = {c.O: c for c in cs}
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(self.U)
        self.dag.add_nodes_from(self.V)
        for c in cs:
            self.dag.add_edges_from([(x,c.O) for x in c.I])
        tp_sort = list(nx.topological_sort(self.dag))
        # self.context_order = [x for x in tp_sort if x in U]
        if state_order is None:
            self.state_order = [x for x in tp_sort if x in V]
        else:
            self.state_order = state_order

    def get_state(self, _context: Dict[GSym, float]) -> Tuple[Dict[str, float], np.ndarray]:
        """Get the state of the SCM given a context.
        
        Args:
            _context (Dict[GSym, float]): Context of the SCM, elsewhere refrerred to as u.
        """
        _ret_dict, _ret_list = {}, []
        _subs = dict(_context)
        for _u in _context.keys():
            locals()[_u] = sympy.symbols(_u)
        for _v in self.state_order:
            _v_sym = eval(self.cs_dict[_v].f)
            locals()[_v] = _v_sym
            _v_val = float(_v_sym.evalf(subs=_subs))
            _ret_dict[_v] = _v_val
            _ret_list.append(_v_val)
            _subs[_v] = _v_val
        return _ret_dict, np.array(_ret_list)
    
    def draw(self):
        pos = nx.spring_layout(self.dag)
        nx.draw(self.dag, pos, with_labels=True, node_size=3000, node_color='skyblue')
        plt.show()
    
    def __repr__(self) -> str:
        return f"SCM(U={self.U}, V={self.V}, cs={self.cs_dict}, state_order={self.state_order})"
    

def test_system_scm():
    a_sp = ComponentOrEquation(['a'], 'A', 'a')
    b_sp = ComponentOrEquation(['b'], 'B', 'b')
    c_sp = ComponentOrEquation(['c', 'A', 'B'], 'C', 'c+A*B')
    a_im = ComponentOrEquation(['a'], 'A', 'a+10')
    b_im = ComponentOrEquation(['b'], 'B', 'b+10')
    c_im = ComponentOrEquation(['c', 'A', 'B'], 'C', 'c+A*B+10')

    S = System([a_sp, b_sp, c_sp])
    T = System([a_im, b_im, c_im])
    M = S.induced_scm()
    N = T.induced_scm(state_order=M.state_order)
    N.draw()

    assert S.is_compatible(T)
    s, _ = M.get_state({'a': 10, 'b': 10, 'c': 10})
    assert s == {'A': 10, 'B': 10, 'C': 110} 

    t, _ = N.get_state({'a': 10, 'b': 10, 'c': 10})
    assert t == {'A': 20, 'B': 20, 'C': 420}

    P = T.get_replacement({'A': a_sp}).induced_scm(state_order=M.state_order)
    t2, _ = P.get_state({'a': 10, 'b': 10, 'c': 10})
    assert t2 == {'A': 10, 'B': 20, 'C': 220}

    Q = T.get_replacement({'A': a_sp, 'B': b_sp}).induced_scm(state_order=M.state_order)
    t3, _ = Q.get_state({'a': 10, 'b': 10, 'c': 10})
    assert t3 == {'A': 10, 'B': 10, 'C': 120}

    print("All tests passed!")
    

if __name__ == "__main__":
    test_system_scm()