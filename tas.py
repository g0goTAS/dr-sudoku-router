from solver import solve
from tsp import puzzle_to_string
import pickle
import inputGenerator as ig


if __name__=='__main__':

    all_inputs = """[Input]
    LogKey:#Tilt X|Light Sensor|Tilt Y|Tilt Z|Up|Down|Left|Right|Start|Select|B|A|L|R|Power|"""
    all_inputs = ''
    with open('paths.pkl', 'rb') as f:
        paths = pickle.load(f)
    with open('puzzles.pkl', 'rb') as f:
        puzzles = pickle.load(f)
    for puzzle_idx in sorted(paths):
        grid = puzzle_to_string(puzzles[puzzle_idx])
        solution = solve(grid)
        next_level = puzzle_idx[1] == 50
        level_inputs = ig.generateInputs(solution,
                                         paths[puzzle_idx],
                                         next_level=next_level)
        all_inputs += level_inputs
        if puzzle_idx > (2,2):
            break
    # all_inputs += '[/Input]'
    with open('Input_Log.txt', 'w') as f:
        f.write(all_inputs[:-1])