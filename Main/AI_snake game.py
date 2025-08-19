import pygame
from pygame import display, time, draw, QUIT, init, KEYDOWN, K_a, K_s, K_d, K_w, K_SPACE, K_r, K_1, K_2, K_3, font
from random import randint, choice
import numpy as np
from math import sin, cos, pi
import json
import os

init()
pygame.font.init()

# Game constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
GRAY = (128, 128, 128)
GOLD = (255, 215, 0)

cols = 25
rows = 25
width = 800
height = 700  # Extra space for UI
game_height = 600
wr = width / cols
hr = game_height / rows

# Game state
class GameState:
    def __init__(self):
        self.score = 0
        self.high_score = self.load_high_score()
        self.level = 1
        self.speed = 8
        self.ai_mode = True
        self.paused = False
        self.game_over = False
        self.power_ups = []
        self.particles = []
        self.difficulty = "Normal"
        self.snake_color = WHITE
        self.trail_effect = True
        
    def save_high_score(self):
        try:
            with open('snake_scores.json', 'w') as f:
                json.dump({'high_score': self.high_score}, f)
        except:
            pass
    
    def load_high_score(self):
        try:
            with open('snake_scores.json', 'r') as f:
                data = json.load(f)
                return data.get('high_score', 0)
        except:
            return 0

game_state = GameState()

screen = display.set_mode([width, height])
display.set_caption("Enhanced AI Snake - Score: 0")
clock = time.Clock()
font_large = font.Font(None, 36)
font_medium = font.Font(None, 24)
font_small = font.Font(None, 18)

# Particle system for visual effects
class Particle:
    def __init__(self, x, y, color, life=30):
        self.x = x + randint(-5, 5)
        self.y = y + randint(-5, 5)
        self.vx = randint(-3, 3)
        self.vy = randint(-3, 3)
        self.color = color
        self.life = life
        self.max_life = life
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        
    def draw(self):
        alpha = int(255 * (self.life / self.max_life))
        color_with_alpha = (*self.color, alpha)
        size = max(1, int(3 * (self.life / self.max_life)))
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), size)

# Power-up system
class PowerUp:
    def __init__(self, x, y, type_name):
        self.x = x
        self.y = y
        self.type = type_name
        self.lifetime = 300  # frames
        self.pulse = 0
        
    def update(self):
        self.lifetime -= 1
        self.pulse += 0.2
        
    def draw(self):
        pulse_size = 3 + sin(self.pulse) * 2
        color = {
            'speed': YELLOW,
            'slow': CYAN,
            'double': GOLD,
            'shield': PURPLE
        }.get(self.type, WHITE)
        
        draw.rect(screen, color, [
            self.x * wr + (wr - pulse_size * 2) / 2,
            self.y * hr + (hr - pulse_size * 2) / 2,
            pulse_size * 2, pulse_size * 2
        ])

class Spot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.f = 0
        self.g = 0
        self.h = 0
        self.neighbors = []
        self.camefrom = []
        self.obstacle = False
        self.permanent_obstacle = False
        
        # Dynamic obstacle generation based on difficulty
        obstacle_chance = {
            "Easy": 1,
            "Normal": 3,
            "Hard": 7,
            "Extreme": 12
        }.get(game_state.difficulty, 3)
        
        if randint(1, 101) < obstacle_chance:
            self.obstacle = True
            self.permanent_obstacle = True

    def show(self, color, trail_alpha=255):
        if trail_alpha < 255:
            # Create fading trail effect
            fade_color = tuple(int(c * trail_alpha / 255) for c in color)
            draw.rect(screen, fade_color, [self.x*wr+2, self.y*hr+2, wr-4, hr-4])
        else:
            draw.rect(screen, color, [self.x*wr+2, self.y*hr+2, wr-4, hr-4])

    def add_neighbors(self):
        self.neighbors = []
        if self.x > 0:
            self.neighbors.append(grid[self.x - 1][self.y])
        if self.y > 0:
            self.neighbors.append(grid[self.x][self.y - 1])
        if self.x < rows - 1:
            self.neighbors.append(grid[self.x + 1][self.y])
        if self.y < cols - 1:
            self.neighbors.append(grid[self.x][self.y + 1])

# Enhanced A* pathfinding with multiple strategies
def get_path_astar(food_pos, snake_body, strategy="direct"):
    # Reset pathfinding data
    for row in grid:
        for spot in row:
            spot.camefrom = []
            spot.f = spot.g = spot.h = 0
    
    start = snake_body[-1]
    goal = food_pos
    
    openset = [start]
    closedset = []
    
    while openset:
        current = min(openset, key=lambda x: x.f)
        openset.remove(current)
        closedset.append(current)
        
        if current == goal:
            break
            
        for neighbor in current.neighbors:
            if neighbor in closedset or neighbor.obstacle or neighbor in snake_body[:-1]:
                continue
                
            tentative_g = current.g + 1
            
            if neighbor not in openset:
                openset.append(neighbor)
            elif tentative_g >= neighbor.g:
                continue
                
            neighbor.camefrom = current
            neighbor.g = tentative_g
            
            # Different heuristics based on strategy
            if strategy == "direct":
                neighbor.h = abs(neighbor.x - goal.x) + abs(neighbor.y - goal.y)
            elif strategy == "safe":
                # Prefer paths away from walls and snake body
                wall_penalty = 0
                if neighbor.x <= 1 or neighbor.x >= rows-2 or neighbor.y <= 1 or neighbor.y >= cols-2:
                    wall_penalty = 5
                neighbor.h = abs(neighbor.x - goal.x) + abs(neighbor.y - goal.y) + wall_penalty
            elif strategy == "spiral":
                # Encourage spiral movement for longer survival
                center_x, center_y = rows//2, cols//2
                spiral_bonus = abs(neighbor.x - center_x) + abs(neighbor.y - center_y)
                neighbor.h = abs(neighbor.x - goal.x) + abs(neighbor.y - goal.y) - spiral_bonus * 0.1
                
            neighbor.f = neighbor.g + neighbor.h
    
    # Reconstruct path
    if current != goal:
        return []  # No path found
        
    path = []
    while current.camefrom:
        if current.x == current.camefrom.x and current.y < current.camefrom.y:
            path.append(2)  # up
        elif current.x == current.camefrom.x and current.y > current.camefrom.y:
            path.append(0)  # down
        elif current.x < current.camefrom.x and current.y == current.camefrom.y:
            path.append(3)  # left
        elif current.x > current.camefrom.x and current.y == current.camefrom.y:
            path.append(1)  # right
        current = current.camefrom
        
    return path

# Advanced AI behaviors
class AIController:
    def __init__(self):
        self.strategy = "direct"
        self.strategies = ["direct", "safe", "spiral"]
        self.stuck_counter = 0
        self.last_positions = []
        
    def get_next_move(self, snake_body, food_pos):
        # Check if snake is stuck in a loop
        head_pos = (snake_body[-1].x, snake_body[-1].y)
        self.last_positions.append(head_pos)
        if len(self.last_positions) > 10:
            self.last_positions.pop(0)
            
        # If stuck, change strategy
        if len(set(self.last_positions)) < 3:
            self.stuck_counter += 1
            if self.stuck_counter > 20:
                self.strategy = choice([s for s in self.strategies if s != self.strategy])
                self.stuck_counter = 0
                self.last_positions = []
        
        path = get_path_astar(food_pos, snake_body, self.strategy)
        
        if not path:
            # Emergency survival mode - just avoid immediate death
            current = snake_body[-1]
            for direction in [0, 1, 2, 3]:
                next_x, next_y = current.x, current.y
                if direction == 0: next_y += 1
                elif direction == 1: next_x += 1
                elif direction == 2: next_y -= 1
                elif direction == 3: next_x -= 1
                
                if (0 <= next_x < rows and 0 <= next_y < cols and 
                    not grid[next_x][next_y].obstacle and 
                    grid[next_x][next_y] not in snake_body[:-1]):
                    return direction
            return 0  # Last resort
            
        return path[-1] if path else 0

def create_power_up():
    if randint(1, 100) < 5:  # 5% chance per frame
        while True:
            x, y = randint(0, rows-1), randint(0, cols-1)
            if not grid[x][y].obstacle and grid[x][y] not in snake:
                power_type = choice(['speed', 'slow', 'double', 'shield'])
                return PowerUp(x, y, power_type)

def apply_power_up(power_up, snake_body):
    if power_up.type == 'speed':
        game_state.speed = min(20, game_state.speed + 3)
        spawn_particles(power_up.x * wr, power_up.y * hr, YELLOW, 15)
    elif power_up.type == 'slow':
        game_state.speed = max(3, game_state.speed - 2)
        spawn_particles(power_up.x * wr, power_up.y * hr, CYAN, 15)
    elif power_up.type == 'double':
        game_state.score += 50
        spawn_particles(power_up.x * wr, power_up.y * hr, GOLD, 20)
    elif power_up.type == 'shield':
        # Remove some obstacles
        removed = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j].obstacle and not grid[i][j].permanent_obstacle and removed < 5:
                    grid[i][j].obstacle = False
                    removed += 1
        spawn_particles(power_up.x * wr, power_up.y * hr, PURPLE, 25)

def spawn_particles(x, y, color, count):
    for _ in range(count):
        game_state.particles.append(Particle(x, y, color))

def update_particles():
    game_state.particles = [p for p in game_state.particles if p.life > 0]
    for particle in game_state.particles:
        particle.update()

def draw_particles():
    for particle in game_state.particles:
        particle.draw()

def create_dynamic_obstacles():
    # Add temporary obstacles that appear and disappear
    if randint(1, 200) == 1:  # Rare event
        x, y = randint(2, rows-3), randint(2, cols-3)
        if not grid[x][y].obstacle and grid[x][y] not in snake:
            grid[x][y].obstacle = True
            # Mark as temporary
            grid[x][y].permanent_obstacle = False

def draw_trail_effect(snake_body):
    if not game_state.trail_effect:
        return
        
    for i, segment in enumerate(snake_body[:-3]):  # Don't trail the head
        alpha = max(50, 255 - (len(snake_body) - i) * 8)
        trail_color = tuple(int(c * alpha / 255) for c in game_state.snake_color)
        segment.show(trail_color)

def draw_ui():
    # Background for UI area
    draw.rect(screen, GRAY, [0, game_height, width, height - game_height])
    
    # Score and stats
    score_text = font_medium.render(f"Score: {game_state.score}", True, WHITE)
    high_score_text = font_medium.render(f"High Score: {game_state.high_score}", True, GOLD)
    level_text = font_medium.render(f"Level: {game_state.level}", True, WHITE)
    speed_text = font_medium.render(f"Speed: {game_state.speed}", True, WHITE)
    
    screen.blit(score_text, (10, game_height + 10))
    screen.blit(high_score_text, (10, game_height + 35))
    screen.blit(level_text, (200, game_height + 10))
    screen.blit(speed_text, (200, game_height + 35))
    
    # AI mode indicator
    mode_color = GREEN if game_state.ai_mode else RED
    mode_text = font_medium.render(f"AI Mode: {'ON' if game_state.ai_mode else 'OFF'}", True, mode_color)
    screen.blit(mode_text, (350, game_height + 10))
    
    # Difficulty indicator
    diff_text = font_medium.render(f"Difficulty: {game_state.difficulty}", True, WHITE)
    screen.blit(diff_text, (350, game_height + 35))
    
    # Controls
    controls = [
        "SPACE: Toggle AI/Manual | R: Restart | 1-3: Difficulty | P: Pause",
        "Manual: WASD to move"
    ]
    for i, control in enumerate(controls):
        control_text = font_small.render(control, True, WHITE)
        screen.blit(control_text, (10, game_height + 60 + i * 15))

def draw_game_over():
    overlay = pygame.Surface((width, height))
    overlay.fill(BLACK)
    overlay.set_alpha(128)
    screen.blit(overlay, (0, 0))
    
    game_over_text = font_large.render("GAME OVER", True, RED)
    final_score_text = font_medium.render(f"Final Score: {game_state.score}", True, WHITE)
    restart_text = font_medium.render("Press R to Restart", True, WHITE)
    
    screen.blit(game_over_text, (width//2 - 100, height//2 - 50))
    screen.blit(final_score_text, (width//2 - 80, height//2))
    screen.blit(restart_text, (width//2 - 90, height//2 + 30))

def reset_game():
    global grid, snake, food, ai_controller, direction
    
    # Reset grid
    grid = [[Spot(i, j) for j in range(cols)] for i in range(rows)]
    for i in range(rows):
        for j in range(cols):
            grid[i][j].add_neighbors()
    
    # Reset snake
    start_x, start_y = rows//2, cols//2
    snake = [grid[start_x][start_y]]
    
    # Reset food
    while True:
        food = grid[randint(0, rows-1)][randint(0, cols-1)]
        if not food.obstacle and food not in snake:
            break
    
    # Reset game state
    if game_state.score > game_state.high_score:
        game_state.high_score = game_state.score
        game_state.save_high_score()
    
    game_state.score = 0
    game_state.level = 1
    game_state.speed = 8
    game_state.game_over = False
    game_state.power_ups = []
    game_state.particles = []
    
    ai_controller = AIController()
    direction = 1

# Initialize game objects
grid = [[Spot(i, j) for j in range(cols)] for i in range(rows)]

for i in range(rows):
    for j in range(cols):
        grid[i][j].add_neighbors()

snake = [grid[rows//2][cols//2]]
food = grid[randint(0, rows-1)][randint(0, cols-1)]
ai_controller = AIController()
direction = 1
frame_count = 0

# Main game loop
done = False
while not done:
    clock.tick(game_state.speed)
    frame_count += 1
    
    # Handle events
    for event in pygame.event.get():
        if event.type == QUIT:
            done = True
        elif event.type == KEYDOWN:
            if event.key == K_SPACE:
                game_state.ai_mode = not game_state.ai_mode
            elif event.key == K_r:
                reset_game()
            elif event.key == K_1:
                game_state.difficulty = "Easy"
                reset_game()
            elif event.key == K_2:
                game_state.difficulty = "Normal"
                reset_game()
            elif event.key == K_3:
                game_state.difficulty = "Hard"
                reset_game()
            elif event.key == ord('p'):
                game_state.paused = not game_state.paused
            elif not game_state.ai_mode and not game_state.game_over:
                # Manual controls
                if event.key == K_w and direction != 0:
                    direction = 2
                elif event.key == K_a and direction != 1:
                    direction = 3
                elif event.key == K_s and direction != 2:
                    direction = 0
                elif event.key == K_d and direction != 3:
                    direction = 1
    
    if game_state.paused or game_state.game_over:
        screen.fill(BLACK)
        if game_state.game_over:
            draw_game_over()
        else:
            pause_text = font_large.render("PAUSED", True, WHITE)
            screen.blit(pause_text, (width//2 - 60, height//2))
        draw_ui()
        display.flip()
        continue
    
    # AI decision making
    if game_state.ai_mode:
        direction = ai_controller.get_next_move(snake, food)
    
    # Move snake
    current = snake[-1]
    next_x, next_y = current.x, current.y
    
    if direction == 0:    # down
        next_y += 1
    elif direction == 1:  # right
        next_x += 1
    elif direction == 2:  # up
        next_y -= 1
    elif direction == 3:  # left
        next_x -= 1
    
    # Check boundaries and collisions
    if (next_x < 0 or next_x >= rows or next_y < 0 or next_y >= cols or
        grid[next_x][next_y].obstacle or grid[next_x][next_y] in snake):
        game_state.game_over = True
        continue
    
    snake.append(grid[next_x][next_y])
    current = snake[-1]
    
    # Check food collision
    if current == food:
        game_state.score += 10
        game_state.level = game_state.score // 100 + 1
        
        # Spawn celebration particles
        spawn_particles(food.x * wr + wr//2, food.y * hr + hr//2, GREEN, 20)
        
        # Generate new food
        attempts = 0
        while attempts < 100:
            food = grid[randint(0, rows-1)][randint(0, cols-1)]
            if not food.obstacle and food not in snake:
                break
            attempts += 1
        
        # Increase speed periodically
        if game_state.score % 50 == 0:
            game_state.speed = min(25, game_state.speed + 1)
    else:
        snake.pop(0)
    
    # Check power-up collisions
    for power_up in game_state.power_ups[:]:
        if current.x == power_up.x and current.y == power_up.y:
            apply_power_up(power_up, snake)
            game_state.power_ups.remove(power_up)
    
    # Update power-ups
    for power_up in game_state.power_ups[:]:
        power_up.update()
        if power_up.lifetime <= 0:
            game_state.power_ups.remove(power_up)
    
    # Spawn new power-ups occasionally
    if frame_count % 180 == 0:  # Every 3 seconds at 60fps
        new_power_up = create_power_up()
        if new_power_up:
            game_state.power_ups.append(new_power_up)
    
    # Create dynamic obstacles occasionally
    if frame_count % 300 == 0:  # Every 5 seconds
        create_dynamic_obstacles()
    
    # Remove temporary obstacles occasionally
    if frame_count % 600 == 0:  # Every 10 seconds
        for i in range(rows):
            for j in range(cols):
                if grid[i][j].obstacle and not grid[i][j].permanent_obstacle and randint(1, 3) == 1:
                    grid[i][j].obstacle = False
    
    # Update particles
    update_particles()
    
    # Update display caption
    display.set_caption(f"Enhanced AI Snake - Score: {game_state.score} - Level: {game_state.level}")
    
    # Drawing
    screen.fill(BLACK)
    
    # Draw grid lines (subtle)
    for i in range(rows + 1):
        draw.line(screen, (20, 20, 20), (i * wr, 0), (i * wr, game_height), 1)
    for i in range(cols + 1):
        draw.line(screen, (20, 20, 20), (0, i * hr), (width, i * hr), 1)
    
    # Draw obstacles with pulsing effect
    for i in range(rows):
        for j in range(cols):
            if grid[i][j].obstacle:
                pulse = sin(frame_count * 0.1) * 0.3 + 0.7
                obstacle_color = tuple(int(c * pulse) for c in RED)
                grid[i][j].show(obstacle_color)
    
    # Draw trail effect
    if game_state.trail_effect and len(snake) > 3:
        draw_trail_effect(snake)
    
    # Draw snake body
    for i, segment in enumerate(snake[:-1]):
        if not game_state.trail_effect or i >= len(snake) - 3:
            segment.show(game_state.snake_color)
    
    # Draw snake head with special effect
    head_pulse = sin(frame_count * 0.3) * 20 + 235
    head_color = (min(255, int(head_pulse)), min(255, int(head_pulse)), 255)
    snake[-1].show(head_color)
    
    # Draw food with pulsing effect
    food_pulse = sin(frame_count * 0.2) * 50 + 200
    food_color = (0, min(255, int(food_pulse)), 0)
    food.show(food_color)
    
    # Draw power-ups
    for power_up in game_state.power_ups:
        power_up.draw()
    
    # Draw particles
    draw_particles()
    
    # Draw UI
    draw_ui()
    
    display.flip()

pygame.quit()