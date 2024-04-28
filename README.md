# A Formal Framework for Distribution of Liabilities in Autonomous Systems

This repository is associated with the paper "A Formal Framework for Distribution of Liabilities in Autonomous Systems". It includes a Python module for calculating k-leg liability, located under `src/liab/`, and a demonstration notebook provided in the `examples/` directory.

## Description

This project implements the methodology discussed in our paper, specifically focusing on the calculation of k-leg liability:

> [W]e propose a two-step procedure: (i) identification of the causal components, and (ii) apportionment of liabilities. Step (i) is done in accordance to Definition \ref{def:ebf-iscause}. Step (ii) is carried out by measuring the effect of fixing each component $X$. However, we do not simply replace each component with its specification in isolation and compare the depth of the resulting replacement system' state in the failure set to that of the original state, because this may not reveal its true effect. Instead, we start by identifying all causal sets that include the component $X$. Within each causal set, we measure the effect of the fault in $X$ by fixing all components within the set both with an without $X$, and calculating the difference. We then average these differences across all causal sets, to determine the average effect of fixing $X$. The effects of fixing all other components are calculated in a similar manner, and the resulting values are normalized to sum to one.

More precicily, if $S$ is a specification system, $T$ is an implementation of $S$, $F$ is the failure set, $u$ is a context, $X\in T$, $X'$ is the corresponding component in $S$, and $k \le |T|$, we use the following formula to calculate k-leg liability:
> $\phi^k_X = \omega \frac{1}{|B|} \sum_{K\in B}max(0, depth_d(SCM(T_{K \rightarrow K'})[u],F) - \\
depth_d(SCM(T_{K\cup \{X\} \rightarrow K' \cup {X'}})[u],F))$

(Please refer to the paper for details.)

 The code is structured into a module to facilitate easy usage and integration into other projects.

## Installation
To set up this project, follow these steps:

```bash
python3 -m venv .venv
source venv/bin/activate
git clone https://github.com/kavaryan/liab.git
pip install -r requirements.txt
```

## Usage
To use the functions provided by the module, you can import them into your Python scripts as follows:

```python
from liab.scm import ComponentOrEquation, GSym, System
from liab.failure import ClosedHalfSpace
from liab.k_leg_liab import k_leg_liab

a_sp = ComponentOrEquation(['a'], 'A', 'a')
b_sp = ComponentOrEquation(['b'], 'B', 'b')
c_sp = ComponentOrEquation(['c'], 'C', 'c')
d_sp = ComponentOrEquation(['d', 'A', 'B', 'C'], 'D', 'd+Max(A*B,A*C,B*C)')
a_im = ComponentOrEquation(['a'], 'A', 'a+10')
b_im = ComponentOrEquation(['b'], 'B', 'b+10')
c_im = ComponentOrEquation(['c'], 'C', 'c+8')
d_im = ComponentOrEquation(['d', 'A', 'B', 'C'], 'D', 'd+Max(A*B,A*C,B*C)+10')

S = System([a_sp, b_sp, c_sp, d_sp])
T = System([a_im, b_im, c_im, d_im])

u = {'a': 10, 'b': 10, 'c': 10, 'd': 10}
F = ClosedHalfSpace({'D': (250, 'ge')})

assert S.induced_scm().get_state(u)[0] == {'A': 10, 'B': 10, 'C': 10, 'D': 110}
assert T.induced_scm().get_state(u)[0] == {'A': 20, 'B': 20, 'C': 18, 'D': 420}

liabs = k_leg_liab(T, S, u, F, k=2)
assert np.isclose(liabs['A'], 0.348, atol=0.01)
assert np.isclose(liabs['A'], liabs['B'])
assert np.isclose(liabs['C'], 0.302, atol=0.01)
assert np.isclose(liabs['D'], 0, atol=0.01)
```

Explore the demonstration examples in the [Example notebook](examples/paper-example.ipynb).

## Documentation
The functions within the [module](src/liab/) are documented. Also, you can explore the examples in the [Example notebook](examples/paper-example.ipynb).

## License
This project is licensed under the [MIT License](LICENSE).