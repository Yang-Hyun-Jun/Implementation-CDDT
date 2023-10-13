# Implementation: CDDT

My implementation code of CDDT algorithm
*Constrained Dirichlet Distribution Trader*

# Overview

- CDDT is an extension of the DDT algorithm in the context of Constrained Markov Decision Processes (CMDP).
-   The constrained portfolio optimization problem is solved in CMDP using Lagrange relaxation.
-   To improve the use efficiency of the violation sample, the objective function of PPO is modified to use an off-policy learning style.

```python
# package 1
pip install pmenv
```