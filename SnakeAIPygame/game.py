import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
from settings import Settings

# allows tuples of points (x-y coordinates) to be easily created
Point = namedtuple('Point', ('x', 'y'))


# enum for direction of snake, to make code more readable
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGameAI:
    """
    Class that controls game logic and graphics
    """
    def __init__(self, width=640, height=480):
        pygame.init()
        self.settings = Settings()
        # init display
        self.display = pygame.display.set_mode((self.settings.screen_width, self.settings.screen_height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """
        Sets snake to initial length and starting position (e.g. after dying, or starting a new game)
        """
        # init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.settings.screen_width / 2, self.settings.screen_height / 2)
        self.snake = [self.head,
                      Point(self.head.x - self.settings.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * self.settings.BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0    # a way to measure time, to punish sake for entering loops

    def _place_food(self):
        """
        Sets the coordinates of the food randomly
        """
        # set x and y coordinate of food
        x = random.randint(0, (self.settings.screen_width - self.settings.BLOCK_SIZE)//self.settings.BLOCK_SIZE)\
            * self.settings.BLOCK_SIZE
        y = random.randint(0, (self.settings.screen_height - self.settings.BLOCK_SIZE) // self.settings.BLOCK_SIZE)\
            * self.settings.BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Updates elements of the game after each frame (e.g. moving the snake, checking if snake collided, etc.)
        """
        self.frame_iteration += 1
        # create event loop and check for quit input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # moves the snake by adding head to front. if snake eats food, the tail won't be removed
        self._move(action)
        self.snake.insert(0, self.head)

        # check if game over (e.g. snake collides with itself or edges, or the game times out)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * int(pow(len(self.snake), 0.8)):
            # if the snake hit itself
            if self.is_collision() and self.head in self.snake[1:]:
                print('\n--Snake hit itself--')
                game_over = True
                # exponentially scale punishment for hitting itself to make it more cautious while with a longer body
                reward = - int(self.settings.reward * pow(len(self.snake), 0.2)) - int(self.settings.reward * 1.0)
                return reward, game_over, self.score
            # if snake hits wall or timed out
            print('\n--Snake hit wall or timed out--')
            game_over = True
            reward = - int(self.settings.reward * 1.0)
            return reward, game_over, self.score

        # if the snake eats the food
        if self.head == self.food:
            self.score += 1
            # reward for eating the food
            reward = self.settings.reward * 1.0 + pow(len(self.snake), 0.05)
            self._place_food()
        else:
            self.snake.pop()

        # update ui and clock
        self._update_ui()
        self.clock.tick(self.settings.SPEED)

        # return game over and score
        game_over = False
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        :return: boolean value for if snake hits itself or the boundary
        """
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.settings.screen_width - self.settings.BLOCK_SIZE or pt.x < 0 or pt.y > self.settings.screen_height - self.settings.BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        # no collision
        return False


    def _update_ui(self):
        """
        Updates positions of all pygame.rect(), images, and text, draws them to the display, then flips display.
        """
        self.display.fill(self.settings.GREEN_DARK)

        for pt in self.snake:
            pygame.draw.rect(self.display, self.settings.GREEN_NEON, pygame.Rect(pt.x, pt.y, self.settings.BLOCK_SIZE, self.settings.BLOCK_SIZE))
            INNER_BLOCK_SIZE = int(0.8 * self.settings.BLOCK_SIZE)
            BLOCK_DIFFERENCE = self.settings.BLOCK_SIZE - INNER_BLOCK_SIZE
            pygame.draw.rect(self.display, self.settings.GREEN, pygame.Rect(pt.x+(BLOCK_DIFFERENCE), pt.y+(BLOCK_DIFFERENCE), INNER_BLOCK_SIZE - BLOCK_DIFFERENCE, INNER_BLOCK_SIZE - BLOCK_DIFFERENCE))

        # draw the food onto the display at the food location
        pygame.draw.rect(self.display, self.settings.PINK_NEON, pygame.Rect(self.food.x, self.food.y, self.settings.BLOCK_SIZE, self.settings.BLOCK_SIZE))

        # draw the text at position (0, 0)
        text = self.settings.font.render("Score: " + str(self.score), True, self.settings.WHITE)
        self.display.blit(text, [0, 0])

        # display all newly drawn in this frame
        pygame.display.flip()

    def _move(self, action):
        """
        Moves the snake's (x, y) position based on an action of 3 booleans (straight, right, left).
        """
        # action = [straight, right, left] e.g. [0, 1, 0] turns the head 90 degrees right.

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        new_dir = None

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        elif np.array_equal(action, [0, 0, 1]):
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.settings.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= self.settings.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= self.settings.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += self.settings.BLOCK_SIZE

        self.head = Point(x, y)
