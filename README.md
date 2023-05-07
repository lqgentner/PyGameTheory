# PyGameTheory

## Preconditions

Please make sure that the packages `numpy`, `scipy`, and `tabulate`
are installed:

``` batch
pip install --upgrade numpy scipy tabulate
```

## Import the module

``` python
import numpy as np
import pygametheory as pgt
```

## Creating a cooperative game

As an example for the cost distribution in a cooperative game, we create
the “petrol supply game” from the lecture with the following value
function:

- $c(A) = 2000$
- $c(B) = 4480$
- $c(C) = 4480$
- $c(\{A, B\}) = 6000$
- $c(\{A, C\}) = 6480$
- $c(\{B, C\}) = 7660$
- $c(\{A, B, C\}) = 9660$

``` python
players = "ABC"
val_func = [2000, 4480, 4480,
            6000, 6480, 7660,
            9660]

coop_game = pgt.CoopGame(players, val_func)
print(coop_game)
```

``` console
A cooperative game with the following value function:
Coalition      Value
-----------  -------
A               2000
B               4480
C               4480
A, B            6000
A, C            6480
B, C            7660
A, B, C         9660
```

Let's find the Shapley values:

``` python
coop_game.shapley()
```

``` console
Coa.     Player      Indv. cost c    Shapley val. φ
-------  --------  --------------  ----------------
A        A                2000.00           2000.00
B        B                4480.00           4480.00
C        C                4480.00           4480.00
-------  --------  --------------  ----------------
A, B     A                2000.00           1760.00
         B                4480.00           4240.00
A, C     A                2000.00           2000.00
         C                4480.00           4480.00
B, C     B                4480.00           3830.00
         C                4480.00           3830.00
-------  --------  --------------  ----------------
A, B, C  A                2000.00           1920.00
         B                4480.00           3750.00
         C                4480.00           3990.00
```

We can also print the Harsanyi coefficients:

``` python
coop_game.harsanyi()
```

``` console
Coalition      Harsanyi λ
-----------  ------------
A                 2000.00
B                 4480.00
C                 4480.00
A, B              -480.00
A, C                 0.00
B, C             -1300.00
A, B, C            480.00
```

## Creating a buying group game

We create a buying group game with the following properties:

- The players are called A, B, and C
- The players order the following item quantities each:
  - Player A: 1000
  - Player B: 1500
  - Player C: 2000
- The item base price is 1.00 €
- The manufacturer gives the following bulk discounts:
  - ≥ 1000: 10 %
  - ≥ 2000: 15 %
  - ≥ 3000: 20 %

``` python
players = "ABC"
units = [1000, 1500, 2000]
discounts = {3000: 0.2,
             2000: 0.15,
             1000: 0.1}
base_price = 1.0

buygrp = pgt.BuyingGroup(players, units, discounts, base_price)
print(buygrp)
```

```console
A buying group game with the following value functions:
Coalition      Cost    Saving
-----------  ------  --------
A               900       100
B              1350       150
C              1700       300
A, B           2125       375
A, C           2400       600
B, C           2800       700
A, B, C        3600       900
```

We print the cost and saving value functions:

``` python
buygrp.coalition_cost()
```

```console
Coalition      Og. cost    Discount     Cost    Saving
-----------  ----------  ----------  -------  --------
A               1000.00        0.10   900.00    100.00
B               1500.00        0.10  1350.00    150.00
C               2000.00        0.15  1700.00    300.00
A, B            2500.00        0.15  2125.00    375.00
A, C            3000.00        0.20  2400.00    600.00
B, C            3500.00        0.20  2800.00    700.00
A, B, C         4500.00        0.20  3600.00    900.00
```

We calculate the proportional costs and savings per player for all
coalitions:

``` python
buygrp.proportional()
```

``` console
Coa.     Player      Indv. cost    Cost π_c    Save π_s    π_c + π_s
-------  --------  ------------  ----------  ----------  -----------
A        A               900.00      900.00      100.00      1000.00
B        B              1350.00     1350.00      150.00      1500.00
C        C              1700.00     1700.00      300.00      2000.00
-------  --------  ------------  ----------  ----------  -----------
A, B     A               900.00      850.00      150.00      1000.00
         B              1350.00     1275.00      225.00      1500.00
A, C     A               900.00      830.77      150.00       980.77
         C              1700.00     1569.23      450.00      2019.23
B, C     B              1350.00     1239.34      233.33      1472.68
         C              1700.00     1560.66      466.67      2027.32
-------  --------  ------------  ----------  ----------  -----------
A, B, C  A               900.00      820.25      163.64       983.89
         B              1350.00     1230.38      245.45      1475.83
         C              1700.00     1549.37      490.91      2040.28
```

## Creating a normal-form game

We use the [Prioner’s
dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma) as an
example:

``` python
A = np.array([[-1, -5], [0, -2]])
B = A.transpose()

prisoners = pgt.NormFormGame(A, B)
print(prisoners)
```

``` console
A normal form game with payoff matrices:
Player A (Row player):
[[-1 -5]
[ 0 -2]]
Player B (Column player):
[[-1  0]
[-5 -2]]
```

Finding the dominant strategies:

``` python
prisoners.dominance()
```

``` console
Found dominant strategies:
Player      Index  Values
--------  -------  --------
A (rows)        2  [ 0 -2]
B (cols)        2  [ 0 -2] 
```

Finding the Nash equilibria:

``` python
prisoners.nash()
```

``` console
Found Nash equilibria:
  No.  Pos.      Val. A    Val. B
-----  ------  --------  --------
    1  (2, 2)        -2        -2 
```
