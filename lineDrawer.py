import pygame
from pygame.locals import *

from inputGenerator import generateInputs


def drawLines(display, buttons, routedButtons):
    for button in buttons:
        if button.text and (button.index not in routedButtons or button.index == routedButtons[-1]):
            filteredButtons = list(filter(lambda b: b.index != button.index and b.text and b.index not in routedButtons, buttons))
            if filteredButtons:
                minDist = min(map(lambda b: len(generateInputs(buttons, [button.index, b.index])) - len(
                    generateInputs(buttons, [button.index])), filteredButtons))
                bestButtons = list(
                    (filter(lambda b: len(generateInputs(buttons, [button.index, b.index])) - len(
                        generateInputs(buttons, [button.index])) == minDist,
                            filteredButtons)))
                for bestButton in bestButtons:
                    pygame.draw.line(display, Color(0, 0, 0), (button.rect.center[0], button.rect.center[1]),
                                     (bestButton.rect.center[0], bestButton.rect.center[1]))
