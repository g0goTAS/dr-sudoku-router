from solver import solve
from tsp import puzzle_to_string
import pickle
import inputGenerator as ig

with open('paths.pkl', 'rb') as f:
    paths = pickle.load(f)
with open('paths_4opt_old.pkl', 'rb') as f:
    paths_old = pickle.load(f)
with open('puzzles.pkl', 'rb') as f:
    puzzles = pickle.load(f)

with open('Input_Log_Manual.txt', 'r') as f:
    manual_data = f.read()
level_split = ig.A + ig.NOTHING * 129 + ig.A + ig.NOTHING * 2
level_trim = ig.NOTHING * 3 + ig.A + ig.NOTHING * 73
manual_data = manual_data.split(level_split)

# COMPARE TO OLD TSP
compare_data = {}
for puzzle_idx, idx_list in paths.items():
    if puzzle_idx not in paths_old:
        continue
    grid = puzzle_to_string(puzzles[puzzle_idx])
    solution = solve(grid, verbose=False)
    next_level = puzzle_idx[1] == 50
    old_list = paths_old[puzzle_idx]

    level_inputs = ig.generateInputs(solution,
                                     idx_list,
                                     next_level=next_level)
    level_inputs = level_inputs[:level_inputs.index(level_trim)]

    old_inputs = ig.generateInputs(solution,
                                   old_list,
                                   next_level=next_level)
    old_inputs = old_inputs[:old_inputs.index(level_trim)]
    compare_data[puzzle_idx] = (old_inputs.count('\n'),
                                level_inputs.count('\n'))

for p, (g, l) in compare_data.items():
    if g-l != 0:
        print(p, '\t', g-l, '\t', l, '\t', g)
print('TOTAL SAVED', sum(g-l for (g, l) in compare_data.values()))


# COMPARE TO GOGO MANUAL
compare_data = {}
for i, gogo_input in enumerate(manual_data[1:]):
    puzzle_idx = (i//50+1, i % 50+1)
    if puzzle_idx not in paths:
        continue

    gogo_input = gogo_input[:gogo_input.index(level_trim)]

    grid = puzzle_to_string(puzzles[puzzle_idx])
    solution = solve(grid, verbose=False)
    next_level = puzzle_idx[1] == 50
    level_inputs = ig.generateInputs(solution,
                                     paths[puzzle_idx],
                                     next_level=next_level)
    level_inputs = level_inputs[:level_inputs.index(level_trim)]

    compare_data[puzzle_idx] = (gogo_input.count('\n'),
                                level_inputs.count('\n'))

for p, (g, l) in compare_data.items():
    print(p, '\t', g-l, '\t', l, '\t', g)
print(sum(g-l for (g, l) in compare_data.values()))
