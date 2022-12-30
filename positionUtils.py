from functools import lru_cache


@lru_cache
def getCage(i):
    x = (i // 3) % 3
    y = i // 27
    return x + 3 * y


@lru_cache
def getRow(i):
    return i // 9


@lru_cache
def getColumn(i):
    return i % 9


@lru_cache
def getSpot(column, row):
    return 9 * row + column
