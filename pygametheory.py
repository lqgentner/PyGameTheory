#!/usr/bin/env python
"""
PyGameTheory
------------

A module for the computation of game theory concepts,
made for the UPC ESEIAAT Game Theory class.

"""

import numpy as np
from scipy.special import factorial
import itertools
from tabulate import tabulate, SEPARATING_LINE
from typing import Literal
from numpy.typing import ArrayLike

__author__ = "Luis Gentner"
__version__ = "1.0.0"
__email__ = "luis.quentin.gentner@estudiantat.upc.edu"


class CoopGame:
    """
    A class for a cooperative game

    Parameters
    ----------

    players: str, list, or tuple
        An iterable containing the player names
    val_func: list
        The value function for all player combinations.
        Must be sorted ([1, 2, 3, 12, 13, 23, 123] for 3 players)
        and have 2^(n-1) entries, with n being the number of players.
    """

    def __init__(self,
                 players: str | list[str | int] | tuple[str | int],
                 val_func: list) -> None:

        self.players = list(players)
        # self.n_players = len(self.players)
        self.coalitions = self._generate_coalitions(self.players)
        self._coalitions_str = [str(", ".join(coal))
                                for coal in self.coalitions]
        if len(val_func) != len(self.coalitions):
            raise ValueError(
                "Value function size doesn't match coalition number")
        self._val_func = val_func

        # Create value function dict
        self.val_dict = dict(zip(self.coalitions, self._val_func))
        # Add empty set to dict
        self.val_dict[()] = 0

    def __str__(self) -> str:
        return "A cooperative game with the following value function:\n" + \
            tabulate({"Coalition": self._coalitions_str,
                      "Value": self._val_func}, headers="keys")

    def _generate_coalitions(
            self, players: str | list[str | int] | tuple[str | int]) -> list[float]:
        """Generate all player coalitions"""
        # "ABC" --> [('A',), ('B',), ('C',),
        #            ('A', 'B'), ('A', 'C'), ('B', 'C'),
        #            ('A', 'B', 'C')]
        return [*itertools.chain.from_iterable(
            itertools.combinations(list(players), r+1) for r in range(len(players)))]

    def proportional(self) -> None:
        """Displays the proportional cost per player for all coalitions"""

        self._distribution_table("prop")

    def shapley(self) -> None:
        """Displays the shapley value per player for all coalitions"""

        self._distribution_table("shap")

    def banzhaf(self) -> None:
        """Displays the Banzhof value per player for all coalitions"""

        self._distribution_table("banz")

    def banzhaf_norm(self) -> None:
        """Displays the normalized Banzhof value per player for all coalitions"""

        self._distribution_table("nbanz")

    def _gamma(self, n: ArrayLike, s: ArrayLike) -> ArrayLike:
        n = np.array(n)
        s = np.array(s)
        return factorial(n-s) * factorial(s-1) / factorial(n)

    def _distribution_table(
        self,
        method: Literal["prop", "shap", "banz", "nbanz"]
    ) -> None:
        """Displays the distributed cost per player for all coalitions"""

        if method == "prop":
            func = self._prop
            dist_str = "Prop. cost π"
        elif method == "shap":
            func = self._shap
            dist_str = "Shapley val. φ"
        elif method == "banz":
            func = self._banz
            dist_str = "Banzhaf val. β"
        elif method == "nbanz":
            func = self._nbanz
            dist_str = "Norm Banzhaf val. nβ"
        else:
            raise ValueError("Provided method not found")

        # List to store table output
        table = []
        header = ["Coa.", "Player", "Indv. cost c", dist_str]

        # Only used for table formatting (dividers)
        last_coal_size = 1

        for coal in self.coalitions:
            for i, player in enumerate(coal):
                # Individual player cost
                indv_cost = float(self.val_dict[(player,)])
                # Calculate distributed cost for player in coalition
                dist_cost = func(player, coal, self.val_dict)

                # Remove multiple occurrences of coal for better readability
                if i > 0:
                    coal_str = None
                else:
                    coal_str = str(", ".join(coal))

                row = [coal_str,
                       player,
                       indv_cost,
                       dist_cost]

                # Add separation line when coa size increases
                curr_coal_size = len(coal)
                if curr_coal_size > last_coal_size:
                    table.append(SEPARATING_LINE)
                last_coal_size = curr_coal_size

                table.append(row)

        print(tabulate(table, headers=header, floatfmt=".2f"))

    def _prop(self,
              player: str,
              coal: tuple,
              val_dict: dict[tuple, int]
              ) -> float:
        "Calculates the proportional cost for a player in a coalition"

        # Sum of individual costs for coalition members
        coal_indv_cost = sum(val_dict[(member,)] for member in coal)
        # Proportional player cost
        return val_dict[coal] * val_dict[(player,)] / coal_indv_cost

    def _shap(self,
              player: str,
              coal: tuple,
              val_dict: dict[tuple, int]
              ) -> float:
        "Calculates the Shapley value for a player in a coalition"

        # Store coalitions with player
        w_player = [subcoal for subcoal in self._generate_coalitions(
            coal) if player in subcoal]
        # Remove player from coalitions with player
        wo_player = [tuple(member for member in subcoal if member != player)
                     for subcoal in w_player]
        # Marginal contribution
        w_player_val = np.array([val_dict[subcoal] for subcoal in w_player])
        wo_player_val = np.array([val_dict[subcoal] for subcoal in wo_player])
        marg_cont = w_player_val - wo_player_val
        # Calculate gamma value
        gamma = self._gamma(len(coal), [len(subcoal) for subcoal in w_player])
        # Shapley value
        return np.sum(gamma * marg_cont)

    def _banz(self,
              player: str,
              coal: tuple,
              val_dict: dict[tuple, int]
              ) -> float:
        "Calculates the Banzhaf value for a player in a coalition"

        # Store coalitions with player
        w_player = [subcoal for subcoal in self._generate_coalitions(
            coal) if player in subcoal]
        # Remove player from coalitions with player
        wo_player = [tuple(member for member in subcoal if member != player)
                     for subcoal in w_player]
        # Marginal contribution
        w_player_val = np.array([val_dict[subcoal] for subcoal in w_player])
        wo_player_val = np.array([val_dict[subcoal] for subcoal in wo_player])
        marg_cont = w_player_val - wo_player_val
        # Banzhaf value
        return np.sum(marg_cont) / 2**(len(coal) - 1)

    def _nbanz(self,
               player: str,
               coal: tuple,
               val_dict: dict[tuple, int]
               ) -> float:
        "Calculates the normed Banzhaf value for a player in a coalition"

        # For every norm. β value, the remaining β values of the coalition members
        # have to be calculated - not very efficient, but clean implementation.
        banz_sum = sum(self._banz(member, coal, val_dict) for member in coal)
        return self._banz(player, coal, val_dict) * val_dict[coal] / banz_sum


class BuyingGroup(CoopGame):
    """
    A class for a cooperative buying group game

    Parameters
    ----------

    players: str, list, or tuple
        An iterable containing the player names
    units: list
        The order size per player
    discounts: dict
        Price reduction per units ordered
    base_price: float
        Base price per unit
    """

    def __init__(self,
                 players: str | list[str | int] | tuple[str | int],
                 units: list[int],
                 discounts: dict[int, float],
                 base_price: float
                 ) -> None:
        self.players = list(players)
        if len(units) != len(self.players):
            raise ValueError("Units must have same length as player size.")
        self.units_dict = dict(zip(self.players, units))
        self.disc_dict = discounts
        self.base_price = base_price
        self.coalitions = self._generate_coalitions(self.players)
        self._coalitions_str = [str(", ".join(coal))
                                for coal in self.coalitions]

        self.coalitions = self._generate_coalitions(self.players)
        self._create_val_funcs()

    def __str__(self) -> str:
        return "A buying group game with the following value functions:\n" + \
            tabulate({"Coalition": self._coalitions_str,
                      "Cost": self._cost_func,
                      "Saving": self._save_func},
                     headers="keys")

    def _discount_lookup(self, units: int) -> float:
        """Lookup discount based on units sold"""

        discount = 0.0
        for item in self.disc_dict.items():
            if units >= item[0]:
                discount = item[1]
                break
        return discount

    def _create_val_funcs(self) -> None:
        """Generate the buying group value functions"""

        cost_func = []
        save_func = []
        orig_list = []
        disc_list = []

        for coal in self.coalitions:
            units = sum(self.units_dict[member] for member in coal)
            orig = units * self.base_price
            disc = self._discount_lookup(units)
            cost = orig * (1 - disc)
            save = orig - cost

            cost_func.append(cost)
            save_func.append(save)
            orig_list.append(orig)
            disc_list.append(disc)

        # Original coalition costs
        self._orig_list = orig_list
        # Coalition discounts
        self._disc_list = disc_list
        # Value functions
        self._cost_func = cost_func
        self._save_func = save_func
        # Value function dicts
        self.cost_dict = dict(zip(self.coalitions, self._cost_func))
        self.save_dict = dict(zip(self.coalitions, self._save_func))
        # Add empty set to dicts
        self.cost_dict[()] = 0
        self.save_dict[()] = 0

    def coalition_cost(self) -> None:
        """Displays the common cost for all coalitions"""

        print(tabulate(
            {
                "Coalition": self._coalitions_str,
                "Og. cost": self._orig_list,
                "Discount": self._disc_list,
                "Cost": self._cost_func,
                "Saving": self._save_func
            },
            headers="keys",
            floatfmt=".2f"))

    def _distribution_table(
        self,
        method: Literal["prop", "shap", "banz", "nbanz"]
    ) -> None:
        """Displays the distributed cost per player for all coalitions"""

        if method == "prop":
            func = self._prop
            dist_str = "π"
        elif method == "shap":
            func = self._shap
            dist_str = "φ"
        elif method == "banz":
            func = self._banz
            dist_str = "β"
        elif method == "nbanz":
            func = self._nbanz
            dist_str = "nβ"
        else:
            raise ValueError("Provided method not found")

        # List to store table output
        table = []
        header = ["Coa.",
                  "Player",
                  "Indv. cost",
                  f"Cost {dist_str}_c",
                  f"Save {dist_str}_s",
                  f"{dist_str}_c + {dist_str}_s"]

        # Only used for table formatting (dividers)
        last_coal_size = 1

        for coal in self.coalitions:
            for i, player in enumerate(coal):
                # Individual player cost
                indv_cost = float(self.cost_dict[(player,)])
                # Distributed cost and saving for player in coalition
                dist_cost = func(player, coal, self.cost_dict)
                dist_save = func(player, coal, self.save_dict)

                # Remove multiple occurrences of coal for better readability
                if i > 0:
                    coal_str = None
                else:
                    coal_str = str(", ".join(coal))

                sv_row = [coal_str,
                          player,
                          indv_cost,
                          dist_cost,
                          dist_save,
                          dist_cost + dist_save]

                # Add separation line when coa size increases
                curr_coal_size = len(coal)
                if curr_coal_size > last_coal_size:
                    table.append(SEPARATING_LINE)
                last_coal_size = curr_coal_size

                table.append(sv_row)

        print(tabulate(table, headers=header, floatfmt=".2f"))

# ------ EXAMPLE ------
# Only execute when run as script
if __name__ == "__main__":

    players = "ABC"
    val_func = [2000, 4480, 4480,
                6000, 6480, 7660,
                9660]

    coop_game = CoopGame(players, val_func)
    print(coop_game)
    print("\nShapley values for player across all coalitions:\n")
    coop_game.shapley()