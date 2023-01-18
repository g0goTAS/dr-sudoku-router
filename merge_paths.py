from tsp import TSPSolver
import os
import pickle

tsp = TSPSolver(suffix='adel')
paths = {}
for fil in sorted(os.listdir('paths')):
    if fil not in {'zpaths_4opt_old.pkl'}:
        pass
    with open(f'paths/{fil}', 'rb') as f:
        add_paths = pickle.load(f)
    print(f'Treating {fil}')
    for puzzle_idx, path in add_paths.items():
        if puzzle_idx not in paths:
            paths[puzzle_idx] = path
        else:
            tsp.set_puzzle(puzzle_idx)
            len_bef = tsp.get_length(paths[puzzle_idx])
            len_aft = tsp.get_length(path)
            if len_bef != len_aft:
                pass
                # print(f'Difference found on puzzle {puzzle_idx}: '
                #       f'{len_bef} vs {len_aft}')
            if len_aft < len_bef:
                if fil == 'zpaths_4opt_old.pkl':
                    print(puzzle_idx, len_aft, len_bef)
                paths[puzzle_idx] = path
new_paths = {k: paths[k] for k in sorted(paths)}
paths = new_paths
with open('paths.pkl', 'wb') as f:
    pickle.dump(paths, f)
