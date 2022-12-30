import pygame
from pygame.locals import *

HEIGHT = 30
WIDTH = 92

NUMBER = 'Number'
ROUTE = 'Route'


class ModeButton(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.x, self.y = x, y
        self.rect = Rect(self.x, self.y, WIDTH, HEIGHT)
        self.mode = NUMBER

    def draw(self, surface):
        pygame.draw.rect(surface, Color(255, 255, 255), self.rect, 0)

        text = pygame.font.SysFont('arial', 25).render(str(self.mode), True, Color(0))
        textRect = text.get_rect()
        textRect.center = self.rect.center
        surface.blit(text, textRect)

    def click(self, pos):
        collision = self.rect.collidepoint(pos[0], pos[1])
        if collision:
            self.mode = NUMBER if self.mode == ROUTE else ROUTE
