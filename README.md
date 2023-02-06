# dr-sudoku-router
This tool aims to automatise the TASing of Dr. Sudoku, a game released on the Gameboy Advance.
Made by g0goTBC and Lightmopp.

# How to run the different sections?

## tsp.py
This file calculates the fastest paths for each of the puzzles. It generates a .pxl file containing the fastest paths, and a .txt file containing logs about the results.

Here's the optional parameters that it can take:

### -min
the starting point of the calculation process. Can specify a level (for example: "14") or a specific puzzle (for example: "(14,1)")

### -max
the ending point of the calculation process (inclusive). Can specify a level (for example: "14") or a specific puzzle (for example: "(14,1)")

### -s
A suffix to be added to the name of the output files.
The following command

tsp.py -s "_example"

will generate the 2 output files with the names "paths_example.pxl" and "logs_example.txt"

### -r
specifies the number of random starts that are used. The default value is 100.

## merge_paths.py
This file combines all the path files (with the .pxl extension) in the current directory into one.

## tas.py
Generates the Input_Log.txt contained in bk2 and tasproj files based on the path file.

## router.py
The ooriginal program used by g0goTBC to solve puzzles manually.
