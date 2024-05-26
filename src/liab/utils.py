from itertools import chain, combinations

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