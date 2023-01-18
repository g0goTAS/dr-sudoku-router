"""Generating TAS input file from the TSP solutions.

Make sure to run tsp.py to generate paths.pkl
"""

from solver import solve
from tsp import puzzle_to_string
import pickle
import inputGenerator as ig
from itertools import product
from tqdm import tqdm

if __name__ == '__main__':

    all_inputs = """[Input]
    LogKey:#Tilt X|Tilt Y|Tilt Z|Light Sensor|Up|Down|Left|Right|Start|Select|B|A|L|R|Power|
    """
    all_inputs += ig.NOTHING * 799
    all_inputs += ig.A
    all_inputs += ig.START
    all_inputs += ig.NOTHING * 38
    all_inputs += ig.A
    all_inputs += ig.NOTHING
    all_inputs += ig.A
    all_inputs += ig.NOTHING * 129
    all_inputs += ig.A
    all_inputs += ig.NOTHING * 2
    with open('paths.pkl', 'rb') as f:
        paths = pickle.load(f)
    with open('puzzles.pkl', 'rb') as f:
        puzzles = pickle.load(f)
    for puzzle_idx in tqdm(product(range(1, 21), range(1, 51)), total=1000):
        if puzzle_idx not in paths:
            print(f'Unsolved puzzle {puzzle_idx}')
            break
        grid = puzzle_to_string(puzzles[puzzle_idx])
        solution = solve(grid, verbose=False)
        next_level = puzzle_idx[1] == 50
        level_inputs = ig.generateInputs(solution,
                                         paths[puzzle_idx],
                                         next_level=next_level)
        all_inputs += level_inputs
    # TRIM ENDING
    all_inputs += '[/Input]'
    with open('Input_Log.txt', 'w') as f:
        f.write(all_inputs[:-1])
