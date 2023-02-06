"""Imports sudoku puzzles in machine readable format

Works if screen is top left of desktop on OpenEmu in x2 scaling

run python ocr.py to take a snapshot of the puzzle on screen
and add it to puzzle.pkl
Make sure the puzzle on screen are shown in order.
"""

# import cv2
from mss import mss
import numpy as np
# from matplotlib import pyplot as plt
import pickle
from itertools import product
# import os

"""
# Generate numbers pixel data numbers.pkl from first puzzle
sct = mss()
bounding_box = {'top': 44, 'left': 0, 'width': 480, 'height': 320}
img_grid = np.array(sct.grab(bounding_box))
numbers = {}
for n in range(1, 10):
    numbers[n] = img_grid[17:17+30, 16+32*(n-1):16+32*n-2]
numbers[0] = numbers[5]
numbers[5] = img_grid[49:79, 48:78]
with open('numbers.pkl', 'wb') as f:
    pickle.dump(numbers, f)
"""

with open('numbers.pkl', 'rb') as f:
    numbers = pickle.load(f)

with open('puzzles.pkl', 'rb') as f:
    puzzles = pickle.load(f)


def get_grid():
    sct = mss()
    bounding_box = {'top': 44, 'left': 0, 'width': 480, 'height': 320}
    img_grid = np.array(sct.grab(bounding_box))
    grid = np.zeros((9, 9), dtype=int)
    for x, y in product(range(9), range(9)):
        img = img_grid[17+32*y:47+32*y, 16+32*x:46+32*x]
        matches = [abs(img[4:-4, 4:-4] - numbers[n][4:-4, 4:-4]).sum()
                   for n in range(10)]
        match = np.argmin(matches)
        if matches[match] != 0:
            raise Exception('No perfect matches found.')
        grid[y, x] = match
    return grid


if __name__ == '__main__':
    level = max(puzzles)
    if level[1] == 50:
        level = (level[0]+1, 1)
    else:
        level = (level[0], level[1] + 1)

    grid = get_grid()
    puzzles[level] = grid
    print(f'Added grid_data for level {level}')
    with open('puzzles.pkl', 'wb') as f:
        pickle.dump(puzzles, f)
