import pygame


class Settings:
    """
    A class to store settings used in different files, e.g. BLOCK_SIZE
    """
    def __init__(self):
        # screen
        self.screen_width = 960     # must be multiple of self.BLOCK_SIZE
        self.screen_height = 720    # must be multiple of self.BLOCK_SIZE
        # font
        pygame.font.init()
        self.font = pygame.font.Font('Helvetica.ttc', 15)
        # colors
        self.WHITE = (255, 255, 255)
        self.RED = (200, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BLUE_LIGHT = (100, 150, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN_NEON = (57, 255, 20)
        self.GREEN = (20, 200, 20)
        self.GREEN_DARK = (15, 40, 15)
        self.PINK_NEON = (255, 16, 240)

        # game
        self.SPEED = 100
        self.BLOCK_SIZE = 60

        # model
        self.MAX_MEMORY = 100_000
        self.BATCH_SIZE = 1_000
        self.lr = 0.001

        # reward
        self.reward = 10
