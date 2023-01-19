"""Traveling salesman problem solver."""

import pickle
from itertools import product, combinations, permutations
from inputGenerator import generateInputs
from solver import solve
from positionUtils import getColumn, getRow
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from math import comb
import random
import time
import os
import argparse


def plot_path(idx_list):
    """Show a path TSP solution in console.

    Parameters
    ----------
    idx_list: list of int
        Solution of the TSP.
        The path is a list of the sudoku grid index
        (from 0 to 80) starting from top left and going
        horizontally, then vertically.
    """
    prev_pos = (0, 9)
    for i in idx_list:
        pos = (getColumn(i), 9-getRow(i))
        if abs(pos[0]-prev_pos[0])+abs(pos[1]-prev_pos[1]) > 5:
            plt.plot([pos[0], prev_pos[0]],
                     [pos[1], prev_pos[1]], c='gray', ls='--')
        else:
            plt.plot([pos[0], prev_pos[0]], [pos[1], prev_pos[1]], c='k')
        prev_pos = pos
    plt.show()


def puzzle_to_string(puzzle):
    """Convert a numpy array puzzle to a grid string.

    Parameters
    ----------
    puzzle: 9x9 int numpy.array
        Sudoku puzzle grid in numpy array format.
        Empty squares are represented by 0.

    Returns
    -------
    str
        Sudoku puzzle grid in string format.
        The 81 digits are ordered from the top left,
        first from left to right, then top to bottom.
        Empty squares are represented by '_'
    """
    grid = ''
    for x, y in product(range(9), range(9)):
        num = puzzle[x, y]
        char = str(num)
        if char == '0':
            char = '_'
        grid += char
    return grid


def permute_list(idx_list, nodes):
    """Permute list according to k-opt optimisation.

    Parameters
    ----------
    idx_list: list of int
        The path is a list of the sudoku grid index
        (from 0 to 80) starting from top left and going
        horizontally, then vertically.

    nodes: list of int
        List of vertices to swap.
        This list comes in pairs, which are the sections of idx_list
        to merge together.
        The first element is paired with an unmoving origin.
        i.e.: nodes = [10, 20, 11] will permute idx_list this way:
              take the first 10 elements of the list, then the last 10 elements
              in reverse order -> idx_list[:11] + idx_list[11:20][::-1]

    Returns
    -------
    list of int
        The new permuted idx_list
    """
    new_list = idx_list[:nodes[0]+1]
    for i in range(len(nodes)//2):
        idx1, idx2 = nodes[2*i+1:2*i+3]
        if idx1 > idx2:
            new_list += idx_list[idx2:idx1+1][::-1]
        else:
            new_list += idx_list[idx1:idx2+1]
    return new_list


class TSPSolver():
    """Solver class for the Traveling Salesman Problem (TSP).

    Parameters
    ----------
    suffix: str, optional
        Suffix to add to every local checkpoint file's path
        (paths.pkl and logs.txt).
        Useful for parallel calculations on a single computer.
        Default is empty string ''.

    Attributes
    ----------
    logs_path: str
        Path of the log file of the computation.

    paths_path: str
        Path of the pickle file containing the [paths]
        attributes of the solutions to the TSP

    paths: dict of (int, int): list of int
        key: puzzle index tuple.
             First element is level (between 1 and 20)
             Second element is number (between 1 and 50)
        value: path with minimal length, solution of the TSP.
               The path is a list of the sudoku grid index
               (from 0 to 80) starting from top left and going
               horizontally, then vertically.
        Is imported from local 'paths.pkl' file

    puzzles: dict of (int, int): 9x9 int numpy.array
        key: puzzle index tuple.
             First element is level (between 1 and 20)
             Second element is number (between 1 and 50)
        value: Sudoku puzzle grid in numpy array format.
               Empty squares are represented by 0.

    grid: str
        Sudoku puzzle grid in string format.
        The 81 digits are ordered from the top left,
        first from left to right, then top to bottom.
        Empty squares are represented by '_'

    solution: str
        Solved sudoku puzzle grid.
        Corresponds to the previous [grid] attribute,
        but solved without empty squares.

    distance: N+1 * N numpy array
        Distance matrix between all N nodes
        (empty squares on the sudoku grid)
        First row of the matrix is the distance of all
        N nodes from the origin position of the cursor.
        Following N rows is the symetrical distance matrix
        between all N nodes.

    dict_distance: dict of (int, int): int
        Distance matrix in dictionary format.
        Slightly faster to read.

    node_to_grid: dict of int:int
        Assuming every every square is indexed (from 0 to 80),
        and empty square is indexed (from 0 to the number of
        empty squared), this dictionary is the correspondance from
        the empty square index to the grid index.

    grid_to_node: dict of int:int
        Assuming every every square is indexed (from 0 to 80),
        and empty square is indexed (from 0 to the number of
        empty squared), this dictionary is the correspondance from
        the grid index to the empty square index.

    min_puzzle: (int, int)
        Minimum puzzle index (inclusive) to solve, if specified.
        First element is level (between 1 and 20)
        Second element is number (between 1 and 50).
        Default is (1, 1)

    max_puzzle: (int, int)
        Maximum puzzle index (inclusive) to solve, if specified.
        First element is level (between 1 and 20)
        Second element is number (between 1 and 50)
        Default is (20, 50)

    with_4opt: bool
        If True, will perform a final 4opt step
        on the best 3opt solution.

    puzzle_index: (int, int)
        Puzzle index that is currently being solved.
        First element is level (between 1 and 20)
        Second element is number (between 1 and 50)
    """
    def __init__(self, suffix=''):
        """Init function."""
        self.logs_path = f'logs{suffix}.txt'

        # Importing self.paths if it exists
        self.paths_path = f'paths{suffix}.pkl'
        if os.path.exists(self.paths_path):
            with open(self.paths_path, 'rb') as f:
                self.paths = pickle.load(f)
        else:
            self.paths = {}

        # Importing self.puzzles
        with open('puzzles.pkl', 'rb') as f:
            self.puzzles = pickle.load(f)

    def compute_distance(self):
        """Compute distance matrix of current puzzle.

        Will use self.grid and self.solution to compute
        self.distance
        """
        num_close = {1: {1, 2, 3, 4, 7},
                     2: {1, 2, 3, 5, 8},
                     3: {1, 2, 3, 6, 9},
                     4: {4, 5, 6, 1, 7},
                     5: {4, 5, 6, 2, 8},
                     6: {4, 5, 6, 3, 9},
                     7: {7, 8, 9, 1, 4},
                     8: {7, 8, 9, 2, 5},
                     9: {7, 8, 9, 3, 6}}

        size = self.grid.count('_')
        self.distance = np.zeros((size+1, size), dtype=int)
        i_g = -1
        for i in range(len(self.grid)+1):
            if i != 0 and self.grid[i-1] != '_':
                continue
            i_g += 1
            if i == 0:
                row, column = 0, 0
                close_numbers = num_close[5]
            else:
                row, column = getRow(i-1), getColumn(i-1)
                close_numbers = num_close[int(self.solution[i-1])]

            j_g = -1
            for j in range(len(self.grid)):
                if self.grid[j] != '_':
                    continue
                j_g += 1
                dest_row, dest_col = getRow(j), getColumn(j)

                # Computing square input distance
                ver_dist = min((dest_row-row) % 9, (row-dest_row) % 9)
                hor_dist = min((dest_col-column) % 9, (column-dest_col) % 9)
                square_dist = (ver_dist + hor_dist + abs(ver_dist - hor_dist) -
                               int(ver_dist != hor_dist))

                # Computing number input distance
                num_dist = 1
                if int(self.solution[j]) not in close_numbers:
                    num_dist += 1

                self.distance[i_g, j_g] = square_dist + num_dist

        # Dictionary hashes are quicker to read than numpy arrays
        self.dict_distance = {
            (i, j): self.distance[i, j]
            for i, j in product(range(self.distance.shape[0]),
                                range(self.distance.shape[1]))
            }

    def compute_idx(self):
        """Compute distance matrix of current puzzle.

        Will use self.grid to compute
        self.grid_to_node and self.node_to_grid
        """
        self.node_to_grid = {}
        i_g = -1
        for i, char in enumerate(self.grid):
            if char != '_':
                continue
            i_g += 1
            self.node_to_grid[i_g] = i
        self.grid_to_node = {v: k for k, v in self.node_to_grid.items()}

    def get_nn_path(self):
        """Compute the nearest neighbours solution of the current puzzle.

        Uses self.grid_to_node and self.distance

        Returns
        -------
        list of int
            The path is a list of the sudoku grid index
            (from 0 to 80) starting from top left and going
            horizontally, then vertically.
        """
        idx_list = [0]
        while len(idx_list) <= len(self.node_to_grid):
            for idx in self.distance[idx_list[-1]].argsort():
                if idx+1 not in idx_list:
                    idx_list.append(idx+1)
                    break
        idx_list = [self.node_to_grid[i-1] for i in idx_list[1:]]
        return idx_list

    def get_length(self,
                   idx_list):
        """Compute number of frames of a certain path.

        Parameters
        ----------
        idx_list: list of int
            The path is a list of the sudoku grid index
            (from 0 to 80) starting from top left and going
            horizontally, then vertically.

        Returns
        -------
        int
        """
        return generateInputs(self.solution, idx_list).count("\n")

    def k_opt(self,
              idx_list,
              k,
              lock_end=False,
              only_best=True,
              quick_pick=False,
              verbose=True):
        """Perform k-opt optimisation on a given path solution.

        Parameters
        ----------
        idx_list: list of int
            The path is a list of the sudoku grid index
            (from 0 to 80) starting from top left and going
            horizontally, then vertically.

        k: int
            Number of edges to swap (not including the dangling end).

        lock_end: bool, optional
            If True, will lock end in place unless when the edge
            if the the end itself
            (i.e. effectively moving the end node on (k-1) opt)
            Default is False.

        only_best: bool, optional
            If True, will only take the best k-opt optimisation
            for each pass, and then redo the analysis on the new path.
            If False, will do all compatible optimisations at once
            before redoing the analysis.
            Default is True

        quick_pick: bool, optional
            If True, will apply an optimisation as soon as one is found.
            Default is False

        verbose: bool, optional
            If True, will print progress and updates on screen.

        Returns
        -------
        list of int
            New idx_list
        """
        optis = []
        idx_list = idx_list[:]
        while True:
            # Iterator on all possible edge combinations
            indexes_iter = combinations(range(len(idx_list)), k)
            if verbose:
                indexes_iter = tqdm(indexes_iter, total=comb(len(idx_list), k))
            for indexes in indexes_iter:

                nodes = []
                for idx in indexes:
                    nodes += [idx-1, idx]
                nodes.append(len(idx_list)-1)

                # Computing distance before edge permutations
                dist_before = 0
                for i in range((len(nodes)-1)//2):
                    if nodes[2*i] == -1:
                        idx1 = 0
                    else:
                        idx1 = self.grid_to_node[idx_list[nodes[2*i]]]+1
                    idx2 = self.grid_to_node[idx_list[nodes[2*i+1]]]
                    dist_before += self.dict_distance[idx1, idx2]

                # Number of permutations (not counting direction switch)
                length = len(nodes)//2
                if lock_end:
                    keep_end = indexes[-1] != len(idx_list)-1
                else:
                    keep_end = False
                if keep_end:
                    length -= 1

                first_permut = True
                for permut, direction in product(
                        permutations(range(length)),
                        product(range(2), repeat=length)):

                    # First permutation is the default, no need to compute
                    if first_permut:
                        first_permut = False
                        continue

                    prune_flag = False
                    dist_after = 0
                    if nodes[0] == -1:
                        idx1 = 0
                    else:
                        idx1 = self.grid_to_node[idx_list[nodes[0]]]+1
                    for i, idx in enumerate(permut):
                        if direction[i] == 0:
                            idx2 = self.grid_to_node[idx_list[nodes[2*idx+1]]]
                            dist_after += self.dict_distance[idx1, idx2]
                            if dist_after >= dist_before:
                                prune_flag = True
                                break
                            idx1 = \
                                self.grid_to_node[idx_list[nodes[2*idx+2]]]+1
                        else:
                            idx2 = self.grid_to_node[idx_list[nodes[2*idx+2]]]
                            dist_after += self.dict_distance[idx1, idx2]
                            if dist_after >= dist_before:
                                prune_flag = True
                                break
                            idx1 = \
                                self.grid_to_node[idx_list[nodes[2*idx+1]]]+1

                    # No need to continue if the distance is alrady larger
                    if prune_flag:
                        continue
                    if keep_end:
                        idx2 = self.grid_to_node[idx_list[nodes[-2]]]
                        dist_after += self.dict_distance[idx1, idx2]

                    save = dist_before - dist_after
                    if save > 0:
                        new_nodes = nodes[0:1]
                        for i, idx in enumerate(permut):
                            if direction[i] == 0:
                                new_nodes += nodes[2*idx+1: 2*idx+3]
                            else:
                                new_nodes += nodes[2*idx+1: 2*idx+3][::-1]
                        if keep_end:
                            new_nodes += nodes[-2:]
                        optis.append((new_nodes, save))
                        if quick_pick:
                            break
                if quick_pick and len(optis) > 0:
                    break

            # If no optimisations are found, break the while True loop
            if len(optis) == 0:
                break

            optis = sorted(optis, key=lambda x: -x[-1])
            # Performing the best optimisation of the list,
            # then do the analysis again.
            if only_best:
                optis = optis[:1]

            used_idx = []
            default_list = list(range(len(idx_list)))
            while True:
                opti_found = False
                for nodes, save in optis:
                    if all(n not in used_idx for n in nodes):
                        opti_found = True
                        break
                if not opti_found:
                    break
                used_idx += nodes
                real_nodes = []
                if nodes[0] == -1:
                    nodes = nodes[1:]
                    real_nodes = [-1]
                real_nodes += [default_list.index(n) for n in nodes]
                default_list = permute_list(default_list, real_nodes)
                idx_list = permute_list(idx_list, real_nodes)

                if verbose:
                    print(f'\tPerforming {k}-opt optimisation '
                          f'saving {save} frame(s)')

            if verbose:
                length = self.get_length(idx_list)
                print(f"\tNew total: {length} frames")
            optis = []
        return idx_list

    def random_starts(self, n_thermal=100):
        """Perform random starting position and picks best 3-opt optis.

        Will also add a nearest neighbour start as a first benchmark.

        Will perform 3-opt optimisation on all guesses and return the
        best optimised path.

        Will return

        Parameters
        ----------
        n_thermal: int, optional
            Number of random starting guesses.
            Default is 100.

        Returns
        -------
        list of [list of int]
            List of all optimised paths
            The path is a list of the sudoku grid index
            (from 0 to 80) starting from top left and going
            horizontally, then vertically.

        """
        idx_list = self.get_nn_path()
        for k in range(2, 4):
            idx_list = self.k_opt(idx_list, k, verbose=False)
        optimal_length = self.get_length(idx_list)
        optimal_paths = [idx_list[:]]

        print(f'Random starts on 3-opt')
        for _ in tqdm(range(n_thermal)):
            random.shuffle(idx_list)
            for k in range(2, 4):
                idx_list = self.k_opt(idx_list, k, verbose=False)
            length = self.get_length(idx_list)
            if length == optimal_length:
                optimal_paths.append(idx_list[:])
            if length < optimal_length:
                optimal_length = length
                optimal_paths = [idx_list[:]]
        return optimal_paths

    def set_puzzle(self, puzzle_idx):
        """Set all attributes necessary for a puzzle computation.

        Will set self.grid, self.solution, self.distance,
        self.dict_distance, self.grid_to_node and self.node_to_grid

        Parameters
        ----------
        puzzle_idx: (int, int)
        """
        self.puzzle_idx = puzzle_idx
        self.grid = puzzle_to_string(self.puzzles[puzzle_idx])
        self.solution = solve(self.grid, verbose=False)
        if '_' in self.solution:
            raise Exception('Puzzle not solved')
        self.compute_distance()
        self.compute_idx()

    def solve_next(self):
        """Solves the next available puzzle to solve.

        Will write result in the paths.pkl file.

        Returns
        -------
        bool
            If False, there is no more puzzle to solve.
        """
        compute_time = time.time()
        puzzle_found = False
        for puzzle_idx in product(range(1, 21), range(1, 51)):
            if (puzzle_idx in self.paths or
                    puzzle_idx < self.min_puzzle or
                    puzzle_idx > self.max_puzzle):
                continue
            puzzle_found = True
            break
        if not puzzle_found:
            return False
        print(f'Solving puzzle {puzzle_idx}')

        self.set_puzzle(puzzle_idx)

        # Compute random guesses optimisation
        first_paths = self.random_starts()
        n_random_paths = len(first_paths)

        # Compute final 4-opt optimisation
        idx_list = first_paths[0]
        if self.with_4opt:
            length_3opt = self.get_length(idx_list)
            idx_list = self.k_opt(idx_list, 4, verbose=True)
            diff_4opt = length_3opt - self.get_length(idx_list)
        else:
            diff_4opt = 'X'
        self.paths[puzzle_idx] = idx_list
        compute_time = time.time() - compute_time

        # Logging
        with open(self.logs_path, 'a+') as log_file:
            _ = log_file.write(
                f'{puzzle_idx}: [{n_random_paths}] random found. '
                f'[{diff_4opt}] 4opt saved. '
                f'[{compute_time:.0f}] s\n'
                )

        # Saving new path
        with open(self.paths_path, 'wb') as f:
            pickle.dump(self.paths, f)

        return True

    def solve(self,
              min_puzzle=(1, 1),
              max_puzzle=(20, 50),
              with_4opt=True):
        """Solve the TSP for all specified puzzles.

        min_puzzle: (int, int), optional
            Minimum puzzle index (inclusive) to solve, if specified.
            First element is level (between 1 and 20)
            Second element is number (between 1 and 50).
            Default is (1, 1)

        max_puzzle: (int, int), optional
            Maximum puzzle index (inclusive) to solve, if specified.
            First element is level (between 1 and 20)
            Second element is number (between 1 and 50)
            Default is (20, 50)

        with_4opt: bool, optional
            If True, will perform a final 4opt step
            on the best solution.
            Default is True.
        """
        self.min_puzzle = min_puzzle
        self.max_puzzle = max_puzzle
        self.with_4opt = with_4opt
        solving_flag = True
        while solving_flag:
            solving_flag = self.solve_next()


if __name__ == '__main__':
    # Parser of the code's arguments
    parser = argparse.ArgumentParser(
        description='Parser for TSP solver'
        )
    # The date must correspond to *existing* sorted DBs on the GCP bucket.
    # Normally this code happens at 2AM-UTC on a UTC server.
    parser.add_argument("-s", "--suffix",
                        help='suffix to add to checkpoint files',
                        default='')
    parser.add_argument("-min", "--minimum",
                        help='minimum puzzle number',
                        default='(1, 1)')
    parser.add_argument("-max", "--maximum",
                        help='maximum puzzle number',
                        default='(20, 50)')
    args = parser.parse_args()
    suffix = args.suffix
    min_puzzle = args.minimum
    if isinstance(min_puzzle, str):
        min_puzzle = eval(min_puzzle)
    if isinstance(min_puzzle, int):
        min_puzzle = (min_puzzle, 1)

    max_puzzle = args.maximum
    if isinstance(max_puzzle, str):
        max_puzzle = eval(max_puzzle)
    if isinstance(max_puzzle, int):
        max_puzzle = (max_puzzle, 50)

    print(f'Computing TSP on puzzles {min_puzzle} to {max_puzzle}. '
          f'file_suffix: {suffix}')
    tsp = TSPSolver(suffix=suffix)
    tsp.solve(min_puzzle=min_puzzle,
              max_puzzle=max_puzzle)
