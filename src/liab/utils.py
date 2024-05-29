from itertools import chain, combinations

import numpy as np

def subsets_upto(iterable, k):
    """Generate all subsets up to size k.
    
    Args:
        iterable (iterable): The iterable to generate subsets from.
        k (int): The maximum size of the subsets. Pass -1 to generate all subsets.
    """
    s = list(iterable)
    if k == -1:
        k = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(k+1))

def all_combs(n) -> list[tuple[int]]:
    """Generate all combinations of binary numbers with n digits."""
    ret = [bin(i)[2:].rjust(n, '0') for i in range(2**n)]
    ret = [tuple(int(xi) for xi in x) for x in ret]
    return ret

def all_combs_f(n) -> list[tuple[float]]:
  """Generate all combinations of binary numbers with n digits and convert them to floats."""
  ret = [bin(i)[2:].rjust(n, '0') for i in range(2**n)]
  ret = [tuple(float(xi) for xi in x) for x in ret]
  return ret

def get_rand_vec(n, seed=42):
  """ Get a random Boolean vector. """
  np.random.seed(seed)
  return np.random.choice(a=[0.0, 1.0], size=(1, n))[0]

def get_rand_tt(n, seed=42):
  """ Get a random truth table (i.e., a Boolean function) over n variables. """
  fv = get_rand_vec(2**n, seed=seed)
  ret = dict(zip(all_combs_f(n), list(fv)))
  return ret

def get_rand_tt_f(n, seed=42):
  """ Get a random truth table (i.e., a Boolean function) over n variables. """
  fv = get_rand_vec(2**n, seed=seed)
  ret = dict(zip(all_combs_f(n), list(fv)))
  return ret

def set_tt(G, seed=42):
  """ Set a random truth table at each node of a DAG to create a Boolean SCM. """
  for n in G.nodes:
    preds = list(G.predecessors(n))
    G.nodes[n]["tt"] = get_rand_tt(len(preds)+1, seed)

def test_subsets_upto():
    assert list(subsets_upto([1, 2, 3], 2)) == [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]
    assert list(subsets_upto([1, 2, 3], -1)) == [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

def test_all_combs():
    assert all_combs(2) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all_combs(3) == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

def test_all_combs_f():
    assert all_combs_f(2) == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    assert all_combs_f(3) == [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)]

########################################################################################
# FIXME: continue using https://colab.research.google.com/drive/1wt1lLlVYveAv2fYZ0ZanlZMYpq272OHz#scrollTo=fF_zS5Yn69OU
########################################################################################
if __name__ == "__main__":
    test_subsets_upto()
    test_all_combs()
    test_all_combs_f()
    print("utils.py: All tests passed")

    