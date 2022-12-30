def gridToString(grid):
    formattedGrid = []
    for list in grid:
        if len(list) == 1:
            formattedGrid.append(list[0])
        else:
            formattedGrid.append("_")
    return "".join(formattedGrid)
