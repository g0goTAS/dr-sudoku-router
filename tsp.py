import pickle
from itertools import product, combinations, permutations
from inputGenerator import generateInputs
from solver import solve
from positionUtils import getColumn, getRow
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from math import comb


def computeGraph(grid, solution):
    num_close = {1: {1, 2, 3, 4, 7},
                 2: {1, 2, 3, 5, 8},
                 3: {1, 2, 3, 6, 9},
                 4: {4, 5, 6, 1, 7},
                 5: {4, 5, 6, 2, 8},
                 6: {4, 5, 6, 3, 9},
                 7: {7, 8, 9, 1, 4},
                 8: {7, 8, 9, 2, 5},
                 9: {7, 8, 9, 3, 6}}

    size = grid.count('_')
    graph = np.zeros((size+1, size), dtype=int)
    i_g = -1
    for i in range(len(grid)+1):
        if i != 0 and grid[i-1] != '_':
            continue
        i_g += 1
        if i == 0:
            row, column = 0, 0
            close_numbers = num_close[5]
        else:
            row, column = getRow(i-1), getColumn(i-1)
            close_numbers = num_close[int(solution[i-1])]

        j_g = -1
        for j in range(len(grid)):
            if grid[j] != '_':
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
            if int(solution[j]) not in close_numbers:
                num_dist += 1

            graph[i_g, j_g] = square_dist + num_dist
    return graph


def list_length(idx_list, grid_to_graph, graph):
    dist = 0
    prev_idx = 0
    for i in idx_list:
        idx = grid_to_graph[i]
        dist += graph[idx+1, prev_idx]
        prev_idx = idx
    return dist


def plot_path(idx_list):
    prev_pos = (0, 9)
    for i in idx_list:
        pos = (getColumn(i), 9-getRow(i))
        if abs(pos[0]-prev_pos[0])+abs(pos[1]-prev_pos[1]) > 5:
            plt.plot([pos[0], prev_pos[0]], [pos[1], prev_pos[1]], c='gray', ls='--')
        else:
            plt.plot([pos[0], prev_pos[0]], [pos[1], prev_pos[1]], c='k')
        prev_pos = pos
    plt.show()


def kOpt(idx_list, graph, grid_to_graph, solution, k):
    optis = []
    graph = {(i, j): graph[i, j]
             for i,j in product(range(graph.shape[0]), range(graph.shape[1]))}
    while True:
        for indexes in tqdm(combinations(range(len(idx_list)), k),
                            total=comb(len(idx_list), k)):
            nodes = []
            for idx in indexes:
                nodes += [idx-1, idx]
            nodes.append(len(idx_list)-1)

            dist_before = 0
            for i in range((len(nodes)-1)//2):
                if nodes[2*i] == -1:
                    idx1 = 0
                else:
                    idx1 = grid_to_graph[idx_list[nodes[2*i]]]+1
                idx2 = grid_to_graph[idx_list[nodes[2*i+1]]]
                dist_before += graph[idx1, idx2]

            length = len(nodes)//2
            keep_end = indexes[-1] != len(idx_list)-1
            # Flag to keep end locked
            keep_end = False
            if keep_end:
                length -= 1

            first_permut = True
            for permut, direction in product(
                    permutations(range(length)),
                    product(range(2), repeat=length)):
                if first_permut:
                    first_permut = False
                    continue
                prune_flag = False
                dist_after = 0
                if nodes[0] == -1:
                    idx1 = 0
                else:
                    idx1 = grid_to_graph[idx_list[nodes[0]]]+1
                for i, idx in enumerate(permut):
                    if direction[i] == 0:
                        idx2 = grid_to_graph[idx_list[nodes[2*idx+1]]]
                        dist_after += graph[idx1, idx2]
                        if dist_after >= dist_before:
                            prune_flag = True
                            break
                        idx1 = grid_to_graph[idx_list[nodes[2*idx+2]]]+1
                    else:
                        idx2 = grid_to_graph[idx_list[nodes[2*idx+2]]]
                        dist_after += graph[idx1, idx2]
                        if dist_after >= dist_before:
                            prune_flag = True
                            break
                        idx1 = grid_to_graph[idx_list[nodes[2*idx+1]]]+1
                if prune_flag:
                    continue
                if keep_end:
                    idx2 = grid_to_graph[idx_list[nodes[-2]]]
                    dist_after += graph[idx1, idx2]

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

        if len(optis) == 0:
            break
        nodes, save = sorted(optis, key = lambda x:-x[-1])[0]
        # TO-DO: We could use ALL compatible ones instead of the best each time (6-9)
        # Or pick the first we find by searching from longer indexes first
        print(f'\tSaving {save} frame(s)')
        new_list = idx_list[:nodes[0]+1]
        for i in range(len(nodes)//2):
            idx1, idx2 = nodes[2*i+1:2*i+3]
            if idx1 > idx2:
                new_list += idx_list[idx2:idx1+1][::-1]
            else:
                new_list += idx_list[idx1:idx2+1]
        idx_list = new_list[:]
        print("\tNew total: {} frames".format(generateInputs(solution, idx_list).count("\n")))
        optis = []
    return idx_list

def solvePath(grid, max_k=4):
    solution = solve(grid)
    if '_' in solution:
        raise Exception('Puzzle not solved')
    graph = computeGraph(grid, solution)

    graph_to_grid = {}
    i_g = -1
    for i, char in enumerate(grid):
        if char != '_':
            continue
        i_g += 1
        graph_to_grid[i_g] = i
    grid_to_graph = {v: k for k, v in graph_to_grid.items()}

    idx_list = [0]
    while len(idx_list) <= len(graph_to_grid):
        for idx in graph[idx_list[-1]].argsort():
            if idx+1 not in idx_list:
                idx_list.append(idx+1)
                break
    idx_list = [graph_to_grid[i-1] for i in idx_list[1:]]

    for k in range(2, max_k+1):
        idx_list = kOpt(idx_list, graph, grid_to_graph, solution, k)
    return idx_list


def puzzle_to_string(puzzle):
    grid = ''
    for x, y in product(range(9), range(9)):
        num = puzzle[x, y]
        char = str(num)
        if char == '0':
            char = '_'
        grid += char
    return grid


if __name__=='__main__':
    with open('paths.pkl', 'rb') as f:
        paths = pickle.load(f)
    with open('puzzles.pkl', 'rb') as f:
        puzzles = pickle.load(f)
    for puzzle_idx in product(range(1, 21), range(1, 50)):
        if puzzle_idx in paths:
            continue
        print(f'Computing path for puzzle {puzzle_idx}.')
        grid = puzzle_to_string(puzzles[puzzle_idx])
        try:
            idx_list = solvePath(grid)
        except Exception as e:
            print(e)
            continue
        paths[puzzle_idx] = idx_list
        with open('paths.pkl', 'wb') as f:
            pickle.dump(paths, f)
        # plot_path(idx_list)