from positionUtils import getColumn, getRow

A = '|    0,    0,    0,    0,.......A...|\n'
R = '|    0,    0,    0,    0,.........r.|\n'
START = '|    0,    0,    0,    0,....S......|\n'
LEFT = '|    0,    0,    0,    0,..L........|\n'
RIGHT = '|    0,    0,    0,    0,...R.......|\n'
UP = '|    0,    0,    0,    0,U..........|\n'
DOWN = '|    0,    0,    0,    0,.D.........|\n'
NOTHING = '|    0,    0,    0,    0,...........|\n'


def generateInputsFromButtons(buttons, routedButtonIndexes):
    grid = ""
    for button in buttons:
        grid += button.text if button.text else "_"
    return generateInputs(grid, routedButtonIndexes)

def generateInputs(grid, routedButtonIndexes, next_level=False):
    currentLocation = (0, 0)
    currentNumber = 5
    inputs = ''
    if not routedButtonIndexes:
        return inputs
    for index in routedButtonIndexes:
        indexInt = int(index)
        currNumber = grid[indexInt]
        x, y = getColumn(indexInt), getRow(indexInt)
        nextLocation = (x, y)
        inputs += getInputsToNextSquare(currentLocation, nextLocation)
        inputs += getInputsToChooseNextNumber(currentNumber, int(currNumber))
        currentNumber = int(currNumber)
        currentLocation = nextLocation

    inputs += NOTHING * 3
    inputs += A
    inputs += NOTHING * 73
    if next_level:
        inputs += R
    inputs += A
    inputs += NOTHING * 129
    inputs += A
    inputs += NOTHING * 2
    return inputs


def getInputsToChooseNextNumber(currentNumber, nextNumber):
    inputs = ''
    currentY, currentX = divmod(currentNumber - 1, 3)
    nextY, nextX = divmod(nextNumber - 1, 3)
    if (nextX - currentX + 3) % 3 == 1:
        inputs += RIGHT
    if (nextX - currentX + 3) % 3 == 2:
        inputs += LEFT
    if (nextY - currentY + 3) % 3 == 1:
        inputs += DOWN
    if (nextY - currentY + 3) % 3 == 2:
        inputs += UP

    if inputs == '':
        inputs = NOTHING
    inputs += A
    inputs += NOTHING
    return inputs


def getInputsToNextSquare(currentLocation, nextLocation):
    direction = ((nextLocation[0] - currentLocation[0] + 9) % 9, (nextLocation[1] - currentLocation[1] + 9) % 9)
    requiredInputs = []
    if 5 > direction[0] > 0:
        requiredInputs.append([RIGHT, direction[0]])
    if 9 > direction[0] > 4:
        requiredInputs.append([LEFT, 9 - direction[0]])
    if 5 > direction[1] > 0:
        requiredInputs.append([DOWN, direction[1]])
    if 9 > direction[1] > 4:
        requiredInputs.append([UP, 9 - direction[1]])

    inputs = ''
    lastInput = None
    while requiredInputs:
        candidateInputs = list(filter(lambda i: i[0] != lastInput, requiredInputs))
        if candidateInputs:
            sortedInputs = sorted(candidateInputs, key=lambda input: input[1], reverse=True)
            inputs += sortedInputs[0][0]
            lastInput = sortedInputs[0][0]
            sortedInputs[0][1] -= 1
            if sortedInputs[0][1] == 0:
                requiredInputs.remove(sortedInputs[0])
        else:
            inputs += NOTHING
            lastInput = NOTHING
    inputs += A
    return inputs
