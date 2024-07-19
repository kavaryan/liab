import random
import networkx as nx
import numpy as np
import re
from sympy.logic import SOPform
from liab.scm import ComponentOrEquation, System

def all_combs_b(n) -> list[tuple[int]]:
    """Generate all combinations of binary numbers with n digits."""
    ret = [bin(i)[2:].rjust(n, '0') for i in range(2**n)]
    ret = [tuple(bool(int(xi)) for xi in x) for x in ret]
    return ret

def all_combs_f(n) -> list[tuple[float]]:
    """Generate all combinations of binary numbers with n digits and convert them to floats."""
    ret = [bin(i)[2:].rjust(n, '0') for i in range(2**n)]
    ret = [tuple(float(xi) for xi in x) for x in ret]
    return ret

def get_rand_vec_b(n, seed=42):
    """ Get a random Boolean vector. """
    rnd = np.random.RandomState(seed)
    return rnd.choice(a=[0, 1], size=(1, n))[0]

def get_rand_vec_f(n, seed=42):
    """ Get a random Boolean vector. """
    rnd = np.random.RandomState(seed)
    return rnd.choice(a=[0.0, 1.0], size=(1, n))[0]

def get_rand_tt_b(n, seed=42) -> dict:
    """ Get a random truth table (i.e., a Boolean function) over n variables. """
    fv = get_rand_vec_f(2**n, seed=seed)
    ret = dict(zip(all_combs_b(n), list(fv)))
    return ret

def get_rand_tt_f(n, seed=42) -> dict:
    """ Get a random truth table (i.e., a Boolean function) over n variables. """
    fv = get_rand_vec_f(2**n, seed=seed)
    ret = dict(zip(all_combs_f(n), list(fv)))
    return ret

def get_random_system(N, vars_type, seed=42):
    """ Create a random DAG. N = |U \\cup V|, the total number of variables
    """
    G=nx.gnp_random_graph(N,0.5,directed=True,seed=seed)
    G=nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])
    assert nx.is_directed_acyclic_graph(G)
    # U = set(f'u_{n}' for n,d in G.in_degree() if d==0)
    # V = set(f'v_{n}' for n,d in G.in_degree() if d==0)

    cs = []

    def change_not(input_string):
        result = re.sub(r'~(\w+)', r'(not \1)', input_string)
        result = re.sub(r'~\((.*?)\)', r'(not(\1))', result)
        return result

    for n in G.nodes:
        preds = list(G.predecessors(n))
        if len(preds) == 0: # This is an exogenous variable
            continue
        rand_tt = get_rand_tt_b(len(preds), seed)
        minterms = [list(k) for k,v in rand_tt.items() if v]
        vars = [f'x_{k}' for k in preds]
        eq = change_not(str(SOPform(vars, minterms)))
        # eq = str(SOPform(vars, minterms))
        c = ComponentOrEquation(vars, f'x_{n}', eq)
        cs.append(c)

    return System(cs, vars_type)

def rerand_system(S: System, seed=42):
    def change_not(input_string):
        result = re.sub(r'~(\w+)', r'(not \1)', input_string)
        result = re.sub(r'~\((.*?)\)', r'(not(\1))', result)
        return result
    
    new_cs = []
    for c in S.cs:
        rand_tt = get_rand_tt_b(len(c.I), seed)
        minterms = [list(k) for k,v in rand_tt.items() if v]
        vars = [k for k in c.I]
        eq = change_not(str(SOPform(vars, minterms)))
        c = ComponentOrEquation(vars, c.O, eq)
        new_cs.append(c)

    return System(new_cs, S.vars_type)


def get_rand_prop(vars: list, num_syms=3, seed=42):
    def change_not(input_string):
        result = re.sub(r'~(\w+)', r'(not \1)', input_string)
        result = re.sub(r'~\((.*?)\)', r'(not(\1))', result)
        return result
    
    syms = random.sample(vars, k=num_syms)
    rand_tt = get_rand_tt_b(len(syms), seed)
    minterms = [list(k) for k,v in rand_tt.items() if v]
    ret = change_not(str(SOPform(syms, minterms)))
    return ret

def test_all_combs():
    assert all_combs_b(2) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all_combs_b(3) == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

def test_all_combs_f():
    assert all_combs_f(2) == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    assert all_combs_f(3) == [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)]

if __name__ == "__main__":
    SEED = 42
    N = 4
    test_all_combs()
    test_all_combs_f()
    
    S = get_random_system(N, seed=SEED)
    print(S)
    T = rerand_system(S, seed=SEED+1)
    print(T)
    
    print("utils.py: All tests passed")

    

