import pickle
from itertools import product
from solver import solve
from positionUtils import getColumn, getRow
import numpy as np
from matplotlib import pyplot as plt


def computeGraph(grid):
    num_close = {1: {1, 2, 3, 4, 7},
                 2: {1, 2, 3, 5, 8},
                 3: {1, 2, 3, 6, 9},
                 4: {4, 5, 6, 1, 7},
                 5: {4, 5, 6, 2, 8},
                 6: {4, 5, 6, 3, 9},
                 7: {7, 8, 9, 1, 4},
                 8: {7, 8, 9, 2, 5},
                 9: {7, 8, 9, 3, 6}}

    solution = solve(grid)
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



with open('puzzles.pkl', 'rb') as f:
    puzzles = pickle.load(f)

puzzle = puzzles[(2,1)]
grid = ''
for x, y in product(range(9), range(9)):
    num = puzzle[x, y]
    char = str(num)
    if char == '0':
        char = '_'
    grid += char

graph = computeGraph(grid)

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

# Compute-length of idx_list (with graph)


optis = []
while True:
    for i in range(len(idx_list)):
        if i == 0:
            idx_i1 = -1
        else:
            idx_i1 = grid_to_graph[idx_list[i-1]]
        idx_i2 = grid_to_graph[idx_list[i]]

        for j in range(i+2, len(idx_list)):
            idx_j1 = grid_to_graph[idx_list[j-1]]
            idx_j2 = grid_to_graph[idx_list[j]]

            dist_before = graph[idx_i1+1, idx_i2] + graph[idx_j1+1, idx_j2]
            dist_after = graph[idx_i1+1, idx_j1] + graph[idx_i2+1, idx_j2]
            if dist_after < dist_before:
                optis.append((i, j, dist_before-dist_after))

        # Final point special case (1-opt)
        dist_before = graph[idx_i1+1, idx_i2]
        dist_after = graph[idx_i1+1, idx_j2]
        if dist_after < dist_before:
            optis.append((i, -1, dist_before-dist_after))

    optis = sorted(optis, key = lambda x:-x[-1])
    if len(optis) == 0:
        break
    i, j = optis[0][0:2]
    print(f'saved {optis[0][2]} frames')
    idx_list = idx_list[:i] + idx_list[i:j][::-1] + idx_list[j:]
    optis = []


# 3-opt
# no-man-behind algo