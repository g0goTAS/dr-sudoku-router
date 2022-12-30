import pygame
from pygame.locals import *

HEIGHT = 30
WIDTH = 92

NUMBER = 'Number'
ROUTE = 'Route'


class StatelessButton(pygame.sprite.Sprite):
    def __init__(self, x, y, text):
        super().__init__()
        self.x, self.y = x, y
        self.rect = Rect(self.x, self.y, WIDTH, HEIGHT)
        self.text = text

    def draw(self, surface):
        pygame.draw.rect(surface, Color(255, 255, 255), self.rect, 0)

        text = pygame.font.SysFont('arial', 25).render(str(self.text), True, Color(0))
        textRect = text.get_rect()
        textRect.center = self.rect.center
        surface.blit(text, textRect)

    def isClicked(self, pos):
        return self.rect.collidepoint(pos[0], pos[1])
