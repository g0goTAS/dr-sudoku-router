import pygame
from pygame.locals import *

TILE_SIZE = 30


class NumberButton(pygame.sprite.Sprite):
    def __init__(self, index, grid):
        super().__init__()
        self.index = index if grid else None
        self.isGrid = grid
        self.y, self.x = divmod(index, 9) if grid else divmod(index, 3)
        self.rect = Rect(self.x * (TILE_SIZE + 1) + self.x // 3 * 5, self.y * (TILE_SIZE + 1) + self.y // 3 * 5,
                         TILE_SIZE, TILE_SIZE) if grid else Rect(self.x * (TILE_SIZE + 1),
                                                                 self.y * (TILE_SIZE + 1) + 300,
                                                                 TILE_SIZE, TILE_SIZE)
        self.isSelected = False
        self.text = '' if grid else str(index + 1)

    def draw(self, surface, routed = False):
        if routed:
            pygame.draw.rect(surface, Color(0, 255, 0), self.rect, 0)
        else:
            pygame.draw.rect(surface, Color(255, 255, 255), self.rect, 0)
        if self.isSelected:
            pygame.draw.rect(surface, Color(255, 0, 0), self.rect, 1)

        text = pygame.font.SysFont('arial', 25).render(str(self.text), True, Color(0))
        textRect = text.get_rect()
        textRect.center = self.rect.center
        surface.blit(text, textRect)

    def isClicked(self, pos):
        return self.rect.collidepoint(pos[0], pos[1])

    def click(self, newText):
        if self.isGrid:
            if self.text == newText:
                self.text = ''
            else:
                self.text = newText
        self.isSelected = True

