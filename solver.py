from copy import deepcopy

from positionUtils import getColumn, getRow, getCage, getSpot

ALL_NUMBERS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]


def solve(grid, verbose=True):
    potentialNumberList = initialiseRemainingNumberList(grid)
    changeInLastIteration = True
    iterationCount = 0
    while changeInLastIteration and not isSolved(potentialNumberList):
        iterationCount += 1
        potentialNumberList, squareSolveChange = solvePerSquare(potentialNumberList)
        potentialNumberList, rowSolveChange = solvePerRow(potentialNumberList)
        potentialNumberList, columnSolveChange = solvePerColumn(potentialNumberList)
        potentialNumberList, cageSolveChange = solvePerCage(potentialNumberList)
        potentialNumberList, intermediateRowSolveChange = intermediateSolvePerRow(potentialNumberList)
        potentialNumberList, intermediateColumnSolveChange = intermediateSolvePerColumn(potentialNumberList)
        potentialNumberList, xWingRowChange = xWingByRow(potentialNumberList)
        potentialNumberList, xWingColumnChange = xWingByColumn(potentialNumberList)
        potentialNumberList, pointingPairsByRowChange = pointingPairsByRow(potentialNumberList, verbose=verbose)
        potentialNumberList, pointingPairsByColumnChange = pointingPairsByColumn(potentialNumberList, verbose=verbose)
        potentialNumberList, nakedPairsByColumnChange = nakedPairsByColumn(potentialNumberList, verbose=verbose)
        potentialNumberList, nakedPairsByRowChange = nakedPairsByRow(potentialNumberList, verbose=verbose)
        # potentialNumberList, hiddenPairByColumnChange = hiddenPairsByColumn(potentialNumberList)

        changeInLastIteration = squareSolveChange or rowSolveChange or columnSolveChange or cageSolveChange or intermediateRowSolveChange or intermediateColumnSolveChange or xWingRowChange or xWingColumnChange or pointingPairsByRowChange or pointingPairsByColumnChange or nakedPairsByColumnChange or nakedPairsByRowChange # or hiddenPairByColumnChange

        gridString = gridToString(potentialNumberList)
        if not changeInLastIteration and gridString.count("_") != 0 and verbose:
            for i in range(0, 81):
                print(potentialNumberList[i])
        if verbose:
            print("\n" + str(iterationCount))
            print(gridString)
    return gridToString(potentialNumberList)


def gridToString(grid):
    formattedGrid = []
    for list in grid:
        if len(list) == 1:
            formattedGrid.append(list[0])
        else:
            formattedGrid.append("_")
    return "".join(formattedGrid)


def hiddenPairsByColumn(potentialNumberList):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)

    for scanColumn in range(0, 9):
        for number1 in range(0, 8):
            spotsNumber1 = []
            for scanRow in range(0, 9):
                if number1 in solvedPotentialNumberList[getSpot(scanColumn, scanRow)]:
                    spotsNumber1.append(getSpot(scanColumn, scanRow))
            if len(spotsNumber1) != 2:
                continue
            spotsNumber2 = []
            for number2 in range(number1 + 1, 9):
                for scanRow in range(0, 9):
                    if number2 in solvedPotentialNumberList[getSpot(scanColumn, scanRow)]:
                        spotsNumber2.append(getSpot(scanColumn, scanRow))
                if len(spotsNumber2) != 2:
                    continue
                if spotsNumber1 != spotsNumber2:
                    continue

                for eliminationNumber in range(0, 9):
                    if eliminationNumber == number1 or eliminationNumber == number2:
                        continue
                    for eliminationSpot in spotsNumber1:
                        if eliminationNumber in solvedPotentialNumberList[eliminationSpot]:
                            solvedPotentialNumberList.remove(eliminationNumber)
                            change = True
    return solvedPotentialNumberList, change





def nakedPairsByColumn(potentialNumberList, verbose=True):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for scanColumn in range(0, 9):
        for scanRow in range(0, 8):
            spot = getSpot(scanColumn, scanRow)
            if len(solvedPotentialNumberList[spot]) == 2:
                for matchPairRow in range(scanRow + 1, 9):
                    matchSpot = getSpot(scanColumn, matchPairRow)
                    if solvedPotentialNumberList[spot] == solvedPotentialNumberList[matchSpot]:
                        if verbose:
                            print(spot, matchSpot)
                        for eliminationRow in range(0, 9):
                            eliminationSpot = getSpot(scanColumn, eliminationRow)
                            if eliminationSpot not in [spot, matchSpot]:
                                for number in solvedPotentialNumberList[spot]:
                                    if number in solvedPotentialNumberList[eliminationSpot]:
                                        solvedPotentialNumberList[eliminationSpot].remove(number)
                                        change = True
    return solvedPotentialNumberList, change

def nakedPairsByCage(potentialNumberList):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for i in range(0, 80):
        if len(solvedPotentialNumberList[i]) != 2:
            continue
        for j in range(i + 1, 81):
            if len(solvedPotentialNumberList) != 2:
                continue
            if getCage(i) != getCage(j):
                continue
            if solvedPotentialNumberList[i] != solvedPotentialNumberList[j]:
                continue
            if i == j:
                continue
            for k in range(0, 81):
                if k == i or k == j:
                    continue
                if getCage(k) != getCage(i):
                    continue
                for number in solvedPotentialNumberList[i]:
                    if number in solvedPotentialNumberList[k]:
                        solvedPotentialNumberList[k].remove(number)
                        change = True
    return solvedPotentialNumberList, change


def nakedPairsByRow(potentialNumberList, verbose=True):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for scanRow in range(0, 9):
        for scanColumn in range(0, 8):
            spot = getSpot(scanColumn, scanRow)
            if len(solvedPotentialNumberList[spot]) == 2:
                for matchPairColumn in range(scanColumn + 1, 9):
                    matchSpot = getSpot(matchPairColumn, scanRow)
                    if solvedPotentialNumberList[spot] == solvedPotentialNumberList[matchSpot]:
                        if verbose:
                            print(spot, matchSpot)
                        for eliminationColumn in range(0, 9):
                            eliminationSpot = getSpot(eliminationColumn, scanRow)
                            if eliminationSpot not in [spot, matchSpot]:
                                for number in solvedPotentialNumberList[spot]:
                                    if number in solvedPotentialNumberList[eliminationSpot]:
                                        solvedPotentialNumberList[eliminationSpot].remove(number)
                                        change = True
    return solvedPotentialNumberList, change


def pointingPairsByRow(potentialNumberList, verbose=True):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for number in ALL_NUMBERS:
        for scanRow in range(0, 9):
            possibleCages = []
            for scanColumn in range(0, 9):
                if number in solvedPotentialNumberList[getSpot(scanColumn, scanRow)]:
                    currentCage = getCage(getSpot(scanColumn, scanRow))
                    if currentCage not in possibleCages:
                        possibleCages.append(currentCage)
            if len(possibleCages) == 1:
                for i in range(0, 81):
                    if getCage(i) == possibleCages[0] and not getRow(i) == scanRow and number in \
                            solvedPotentialNumberList[i]:
                        if i == 10 and number == 3 and verbose:
                            print("e")
                        solvedPotentialNumberList[i].remove(number)
                        change = True
    return solvedPotentialNumberList, change


def pointingPairsByColumn(potentialNumberList, verbose=True):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for number in ALL_NUMBERS:
        for scanColumn in range(0, 9):
            possibleCages = []
            for scanRow in range(0, 9):
                if number in solvedPotentialNumberList[getSpot(scanColumn, scanRow)]:
                    currentCage = getCage(getSpot(scanColumn, scanRow))
                    if currentCage not in possibleCages:
                        possibleCages.append(currentCage)
            if len(possibleCages) == 1:
                for i in range(0, 81):
                    if getCage(i) == possibleCages[0] and not getColumn(i) == scanColumn and number in \
                            solvedPotentialNumberList[i]:
                        if i == 10 and number == 3 and verbose:
                            print("f")
                        solvedPotentialNumberList[i].remove(number)
                        change = True
    return solvedPotentialNumberList, change


def xWingByRow(potentialNumberList):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for number in ALL_NUMBERS:
        possibleColumnCombinations = []
        for scanRow in range(0, 9):
            currentPossibleColumnCombination = []
            for column in range(0, 9):
                if number in solvedPotentialNumberList[getSpot(column, scanRow)]:
                    currentPossibleColumnCombination.append(column)

            if len(currentPossibleColumnCombination) == 2:
                if currentPossibleColumnCombination in possibleColumnCombinations:  # X wing detected
                    otherRow = possibleColumnCombinations.index(currentPossibleColumnCombination)
                    for eliminationRow in [otherRow, scanRow]:
                        for eliminationColumn in range(0, 9):
                            if number in solvedPotentialNumberList[getSpot(eliminationColumn,
                                                                           eliminationRow)] and eliminationColumn not in currentPossibleColumnCombination:
                                solvedPotentialNumberList[getSpot(eliminationColumn, eliminationRow)].remove(number)
                                change = True
            possibleColumnCombinations.append(currentPossibleColumnCombination)
    return solvedPotentialNumberList, change


def xWingByColumn(potentialNumberList):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for number in ALL_NUMBERS:
        possibleRowCombinations = []
        for scanColumn in range(0, 9):
            currentPossibleRowCombination = []
            for row in range(0, 9):
                if number in solvedPotentialNumberList[getSpot(scanColumn, row)]:
                    currentPossibleRowCombination.append(row)

            if len(currentPossibleRowCombination) == 2:
                if currentPossibleRowCombination in possibleRowCombinations:  # X wing detected
                    otherColumn = possibleRowCombinations.index(currentPossibleRowCombination)
                    for eliminationColumn in [otherColumn, scanColumn]:
                        for eliminationRow in range(0, 9):
                            if number in solvedPotentialNumberList[getSpot(eliminationColumn,
                                                                           eliminationRow)] and eliminationRow not in currentPossibleRowCombination:
                                solvedPotentialNumberList[getSpot(eliminationColumn, eliminationRow)].remove(number)
                                change = True
            possibleRowCombinations.append(currentPossibleRowCombination)
    return solvedPotentialNumberList, change


def intermediateSolvePerRow(potentialNumberList):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for cage in range(0, 9):
        for number in getUnsolvedNumbersInCage(solvedPotentialNumberList, cage):
            possibleRows = []
            for i in range(0, 81):
                if getCage(i) == cage and number in solvedPotentialNumberList[i] and getRow(i) not in possibleRows:
                    possibleRows.append(getRow(i))

            if len(possibleRows) == 1:
                for i in range(0, 81):
                    if getRow(i) == possibleRows[0] and getCage(i) != cage and number in solvedPotentialNumberList[i]:
                        solvedPotentialNumberList[i].remove(number)
                        change = True
    return solvedPotentialNumberList, change


def intermediateSolvePerColumn(potentialNumberList):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for cage in range(0, 9):
        for number in getUnsolvedNumbersInCage(solvedPotentialNumberList, cage):
            possibleColumns = []
            for i in range(0, 81):
                if getCage(i) == cage and number in solvedPotentialNumberList[i] and getColumn(
                        i) not in possibleColumns:
                    possibleColumns.append(getColumn(i))

            if len(possibleColumns) == 1:
                for i in range(0, 81):
                    if getColumn(i) == possibleColumns[0] and getCage(i) != cage and number in \
                            solvedPotentialNumberList[i]:
                        solvedPotentialNumberList[i].remove(number)
                        change = True
    return solvedPotentialNumberList, change


def solvePerCage(potentialNumberList):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for i in range(0, 9):
        unsolvedNumbers = getUnsolvedNumbersInCage(solvedPotentialNumberList, i)
        for unsolvedNumber in unsolvedNumbers:
            possibleSpots = []
            for j in range(0, 81):
                spot = j
                if getCage(j) == i and unsolvedNumber in solvedPotentialNumberList[spot]:
                    possibleSpots.append(spot)
            if len(possibleSpots) == 1:
                solvedPotentialNumberList[possibleSpots[0]] = [unsolvedNumber]
                change = True
    return solvedPotentialNumberList, change


def getUnsolvedNumbersInCage(potentialNumberList, cageNumber):
    unsolvedNumbers = ALL_NUMBERS[::]
    for i in range(0, 81):
        if getCage(i) == cageNumber and len(potentialNumberList[i]) == 1:
            unsolvedNumbers.remove(potentialNumberList[i][0])
    return unsolvedNumbers


def solvePerColumn(potentialNumberList):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for i in range(0, 9):
        unsolvedNumbers = getUnsolvedNumbersInColumn(solvedPotentialNumberList, i)
        for unsolvedNumber in unsolvedNumbers:
            possibleSpots = []

            for j in range(0, 9):
                spot = i + j * 9
                if unsolvedNumber in solvedPotentialNumberList[spot]:
                    possibleSpots.append(spot)
            if len(possibleSpots) == 1:
                solvedPotentialNumberList[possibleSpots[0]] = [unsolvedNumber]
                change = True
    return solvedPotentialNumberList, change


def solvePerRow(potentialNumberList):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for i in range(0, 9):
        unsolvedNumbers = getUnsolvedNumbersInRow(solvedPotentialNumberList, i)
        for unsolvedNumber in unsolvedNumbers:
            possibleSpots = []
            for j in range(0, 9):
                spot = 9 * i + j
                if unsolvedNumber in solvedPotentialNumberList[spot]:
                    possibleSpots.append(spot)
            if len(possibleSpots) == 1:
                solvedPotentialNumberList[possibleSpots[0]] = [unsolvedNumber]
                change = True
    return solvedPotentialNumberList, change


def getUnsolvedNumbersInRow(potentialNumberList, rowNumber):
    unsolvedNumbers = ALL_NUMBERS[::]
    for i in range(9 * rowNumber, 9 * rowNumber + 9):
        if len(potentialNumberList[i]) == 1:
            unsolvedNumbers.remove(potentialNumberList[i][0])
    return unsolvedNumbers


def getUnsolvedNumbersInColumn(potentialNumberList, columnNumber):
    unsolvedNumbers = ALL_NUMBERS[::]
    for i in range(columnNumber, 81 + columnNumber, 9):
        if len(potentialNumberList[i]) == 1:
            unsolvedNumbers.remove(potentialNumberList[i][0])
    return unsolvedNumbers


def solvePerSquare(potentialNumberList):
    change = False
    solvedPotentialNumberList = deepcopy(potentialNumberList)
    for i in range(0, 81):
        if len(solvedPotentialNumberList[i]) > 1:
            newList = getPotentialNumberList(solvedPotentialNumberList, i)
            for number in ALL_NUMBERS:
                if number in potentialNumberList[i] and number not in newList:
                    solvedPotentialNumberList[i].remove(number)
                    change = True
    return solvedPotentialNumberList, change


def getPotentialNumberList(potentialNumberList, currentIndex):
    cage = getCage(currentIndex)
    row = getRow(currentIndex)
    column = getColumn(currentIndex)

    notPossibleNumbers = []
    for i in range(0, 81):
        if getRow(i) == row or getCage(i) == cage or getColumn(i) == column:
            if len(potentialNumberList[i]) == 1:
                notPossibleNumbers.append(potentialNumberList[i][0])

    possibleNumbers = []
    for number in ALL_NUMBERS:
        if number not in notPossibleNumbers:
            possibleNumbers.append(number)
    return possibleNumbers


def isSolved(potentialNumberList):
    for potentialNumbers in potentialNumberList:
        if len(potentialNumbers) > 1:
            return False
    return True


def initialiseRemainingNumberList(grid):
    potentialNumberList = []
    for number in grid:
        if number == "_":
            potentialNumberList.append(ALL_NUMBERS[::])
        else:
            potentialNumberList.append([number])
    return potentialNumberList
