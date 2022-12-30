import pygame
from pygame.locals import *
from pyperclip import copy

import lineDrawer
from button import NumberButton
from inputGenerator import generateInputs
from modeButton import ModeButton
from solver import solve
from statelessButton import StatelessButton

pygame.init()
ScreenWidth = 290
ScreenHeight = 395


def addRerecord():
    recFile = open('rerecord.txt', 'r')
    recCount = int(recFile.read())
    recFile.close()
    recFile = open('rerecord.txt', 'w')
    recFile.write(str(recCount + 1))
    recFile.close()


def manageGridButtonClicks():
    selectedNumber = list(filter(lambda button: button.isSelected, numberButtons))
    selectedNumber = selectedNumber[0].text if len(selectedNumber) else ''
    clickedButton = list(filter(lambda button: button.isClicked(pygame.mouse.get_pos()), gridButtons))
    if len(clickedButton) > 0:
        if modeButton.mode == 'Number':
            for button in gridButtons:
                button.isSelected = False
            clickedButton[0].click(selectedNumber)
        else:
            if clickedButton[0].index not in routedButtonIndexes and clickedButton[0].text != '':
                routedButtonIndexes.append(clickedButton[0].index)
                inputs = generateInputs(gridButtons, routedButtonIndexes)
                print(inputs)
                if len(routedButtonIndexes) == len(list(filter(lambda b: b.text, gridButtons))):
                    copy(inputs)
                    print("copied")
            elif clickedButton[0].index in routedButtonIndexes:
                routedButtonIndexes.remove(clickedButton[0].index)
            for button in gridButtons:
                button.isSelected = button.index in routedButtonIndexes and button.index == routedButtonIndexes[-1]


def manageNumberButtonClicks():
    clickedButton = list(filter(lambda button: button.isClicked(pygame.mouse.get_pos()), numberButtons))
    if len(clickedButton) > 0:
        for button in numberButtons:
            button.isSelected = False
        clickedButton[0].click('')


def generateGridString(gridButtons):
    string = ""
    for button in gridButtons:
        if button.text == "":
            string += "_"
        else:
            string += button.text
    return string


def solveGrid(gridButtons):
    unsolvedGridString = generateGridString(gridButtons)
    solvedGridString = solve(unsolvedGridString)
    if "_" not in solvedGridString:
        for i in range(0, 81):
            if gridButtons[i].text == "":
                gridButtons[i].text = solvedGridString[i]
            else:
                gridButtons[i].text = ""


def manageEvents(events):
    global routedButtonIndexes
    global gridButtons
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            addRerecord()
            manageGridButtonClicks()
            manageNumberButtonClicks()
            modeButton.click(pygame.mouse.get_pos())

            if clearRouteButton.isClicked(pygame.mouse.get_pos()):
                routedButtonIndexes = []
            if clearNumButton.isClicked(pygame.mouse.get_pos()):
                for i in gridButtons:
                    i.text = ''
                routedButtonIndexes = []
            if solveButton.isClicked(pygame.mouse.get_pos()):
                solveGrid(gridButtons)
            draw()


def draw():
    displaySurface.fill(Color(220, 220, 220))
    for button in gridButtons:
        button.draw(displaySurface, button.index in routedButtonIndexes)
    for button in numberButtons:
        button.draw(displaySurface)
    modeButton.draw(displaySurface)
    clearRouteButton.draw(displaySurface)
    clearNumButton.draw(displaySurface)
    solveButton.draw(displaySurface)
    lineDrawer.drawLines(displaySurface, gridButtons, routedButtonIndexes)
    drawInputLength(displaySurface, generateInputs(gridButtons, routedButtonIndexes))
    pygame.display.update()


def drawInputLength(displaySurface, inputs):
    text = pygame.font.SysFont('arial', 25).render(str(inputs.count("\n")), True, Color(0))
    textRect = text.get_rect()
    textRect.center = (248, 310)
    displaySurface.blit(text, textRect)


def initialiseGridButtons():
    buttons = []
    for i in range(0, 81):
        buttons.append(NumberButton(i, True))
    return buttons


def initialiseNumberButtons():
    buttons = []
    for i in range(0, 9):
        buttons.append(NumberButton(i, False))
    return buttons


displaySurface = pygame.display.set_mode((ScreenWidth, ScreenHeight))
pygame.display.set_caption('Sudoku router')
gridButtons = initialiseGridButtons()
numberButtons = initialiseNumberButtons()
modeButton = ModeButton(98, 300)
clearRouteButton = StatelessButton(98, 331, 'Clear path')
clearNumButton = StatelessButton(98, 362, 'Clear num')
solveButton = StatelessButton(198, 331, 'Solve')
routedButtonIndexes = []

draw()
while True:
    manageEvents(pygame.event.get())
