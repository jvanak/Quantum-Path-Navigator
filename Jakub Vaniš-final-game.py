import pygame
import random
import numpy as np
from pygame.locals import (
    K_h, K_x, K_z, K_s, K_ESCAPE, K_f, QUIT
)

SCREEN_WIDTH = 1700
SCREEN_HEIGHT = 900
WORLD_WIDTH = 4000
WORLD_HEIGHT = 3000
INITIAL_OFFSET = 500  # Horizontal offset before tree starts

# Color Definitions
GREEN = (0, 200, 0)
RED = (200, 0, 0)
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)

# Quantum Gate Matrices
H_GATE = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)  # Hadamard gate
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X gate
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli-Z gate
S_GATE = np.array([[1, 0], [0, 1j]], dtype=complex)  # Phase gate

# Pygame Initialization
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 28)
end_font = pygame.font.SysFont("monospace", 72)

SHIP_SPEED = 2  # Movement speed in pixels per frame

class Qubit:
    def __init__(self):
        # Initialize with random Bloch sphere coordinates
        theta = np.arccos(1 - 2 * random.random())
        phi = 2 * np.pi * random.random()
        self.state = np.array([
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2)
        ], dtype=complex)
        self.gate_history = []

    # Apply quantum gates and store history
    def gate_H(self): self.state = H_GATE @ self.state; self.gate_history.append('H')
    def gate_X(self): self.state = X_GATE @ self.state; self.gate_history.append('X')
    def gate_Z(self): self.state = Z_GATE @ self.state; self.gate_history.append('Z')
    def gate_S(self): self.state = S_GATE @ self.state; self.gate_history.append('S')

    def collapse_x_basis(self):
        # Simulate measurement in the X basis and collapse the state
        plus = (self.state[0] + self.state[1]) / np.sqrt(2)
        p_plus = abs(plus) ** 2
        outcome = 0 if random.random() < p_plus else 1
        if outcome == 0:
            self.state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], complex)
        else:
            self.state = np.array([1/np.sqrt(2), -1/np.sqrt(2)], complex)
        return outcome

    def probabilities_in_x_basis(self):
        # Calculate |+> and |-> probabilities
        plus = (self.state[0] + self.state[1]) / np.sqrt(2)
        return abs(plus) ** 2, 1 - abs(plus) ** 2

    def get_bloch_coordinates(self):
        # Return (x, y, z) for Bloch sphere visualization
        alpha, beta = self.state
        x = 2 * (alpha.conjugate() * beta).real
        y = 2 * (alpha.conjugate() * beta).imag
        z = abs(alpha) ** 2 - abs(beta) ** 2
        return x, y, z

    def pretty_state(self):
        # Return readable quantum state string
        def format_amplitude(c):
            mag = abs(c)
            if mag < 0.01:
                return ""
            phase = np.angle(c)
            return f"{mag:.2f}" if np.isclose(phase, 0) else f"{mag:.2f}e^({phase:.2f}i)"
        alpha, beta = self.state
        parts = []
        if format_amplitude(alpha):
            parts.append(f"{format_amplitude(alpha)}|0>")
        if format_amplitude(beta):
            parts.append(f"{format_amplitude(beta)}|1>")
        return " + ".join(parts) if parts else "0"

def generate_tree(depth, start_x, start_y, dx=400, dy=200):
    coords = [(start_x, start_y)]
    paths, forks, path_indices = [], {}, {}
    next_node = 1
    for d in range(depth):
        for i in range(2 ** d):
            parent = sum(2 ** k for k in range(d)) + i
            px, py = coords[parent]
            forks[parent] = []
            for dy_sign in [-1, 1]:  # Two branches for each parent node
                coords.append((px + dx, py + dy * dy_sign))
                paths.append(((px, py), coords[-1]))
                forks[parent].append(next_node)
                next_node += 1
            path_indices[parent] = [len(paths) - 2, len(paths) - 1]
    leaves = [i for i in range(len(coords)) if i not in forks]
    return coords, paths, forks, path_indices, leaves

# Ship Movement
def update_ship():
    global ship_node, ship_phase, ship_x, ship_y, game_over, game_won, qubit

    # Calculate the direction and move
    ex, ey = ship_path_end
    dx, dy = ex - ship_x, ey - ship_y
    dist = (dx ** 2 + dy ** 2) ** 0.5

    if dist < SHIP_SPEED:
        # Arrived at a node
        ship_x, ship_y = ex, ey
        ship.rect.center = (ship_x, ship_y)

        # First node
        if ship_phase == 'initial':
            ship_phase = 'tree'
            result = qubit.collapse_x_basis()
            # If the outcome is |->, give game over
            if result == 1:
                game_over = True
                game_won = False
                return
            # Otherwise, move to the correct child (green branch)
            child = FORKS[0][fork_state[0]]
            ship_node = child
            update_ship_path(coords[0], coords[child])
            qubit = Qubit()
            return

        elif ship_node in FORKS:
            result = qubit.collapse_x_basis()
            if result == 1:
                game_over = True
                game_won = False
                return
            child = FORKS[ship_node][fork_state[ship_node]]
            update_ship_path(coords[ship_node], coords[child])
            ship_node = child
            qubit = Qubit()
            return
        else:
            # Leaf node
            game_over = True
            game_won = True
        return

    # Move toward the target node
    ship_x += SHIP_SPEED * dx / dist
    ship_y += SHIP_SPEED * dy / dist
    ship.rect.center = (ship_x, ship_y)

# Update movement vector for the ship
def update_ship_path(start, end):
    global ship_path_start, ship_path_end
    ship_path_start = start
    ship_path_end = end

def draw_tree(camera_x, camera_y):
    pygame.draw.line(screen, GREEN, (start_line_start[0] - camera_x, start_line_start[1] - camera_y),
                     (coords[0][0] - camera_x, coords[0][1] - camera_y), 4)
    for parent, (i0, i1) in PATH_INDICES.items():
        for outcome, color in [(fork_state[parent], GREEN), (1 - fork_state[parent], RED)]:
            idx = PATH_INDICES[parent][outcome]
            (sx, sy), (ex, ey) = PATHS[idx]
            pygame.draw.line(screen, color, (sx - camera_x, sy - camera_y), (ex - camera_x, ey - camera_y), 4)

# Input Handling
def handle_pressed_keys(pressed_keys, prev_keys):
    if pressed_keys[K_h] and not prev_keys[K_h]:
        qubit.gate_H()
    if pressed_keys[K_x] and not prev_keys[K_x]:
        qubit.gate_X()
    if pressed_keys[K_z] and not prev_keys[K_z]:
        qubit.gate_Z()
    if pressed_keys[K_s] and not prev_keys[K_s]:
        qubit.gate_S()
    if pressed_keys[K_f]:
        pygame.time.wait(10000)  # Pause

# Game State Initialization
DEPTH = 6
root_x, root_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
coords, PATHS, FORKS, PATH_INDICES, LEAVES = generate_tree(DEPTH, root_x, root_y)
start_line_start = (coords[0][0] - INITIAL_OFFSET, coords[0][1])
fork_state = {p: random.choice([0, 1]) for p in FORKS}
qubit = Qubit()

# Create ship sprite
ship = pygame.sprite.Sprite()
ship.surf = pygame.Surface((20, 20), pygame.SRCALPHA)
pygame.draw.circle(ship.surf, WHITE, (10, 10), 10)
ship.rect = ship.surf.get_rect()

# Create goal sprite
goal = pygame.sprite.Sprite()
goal.surf = pygame.Surface((1, 1))  # Will be resized later
goal.rect = goal.surf.get_rect()

# Initialize ship location
ship_node = 0
ship_phase = 'initial'
ship_path_start = start_line_start
ship_path_end = coords[0]
ship_x, ship_y = ship_path_start
ship.rect.center = ship_path_start

game_over = False
game_won = False
prev_keys = pygame.key.get_pressed()

# Main Game Loop
running = True
while running:
    dt = clock.tick(60)
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    pressed_keys = pygame.key.get_pressed()
    if pressed_keys[K_ESCAPE]:
        running = False

    if not game_over:
        handle_pressed_keys(pressed_keys, prev_keys)
        update_ship()

    camera_x = max(0, min(ship_x - SCREEN_WIDTH / 2, WORLD_WIDTH - SCREEN_WIDTH))
    camera_y = ship_y - SCREEN_HEIGHT / 2
    screen.fill(BLACK)
    draw_tree(camera_x, camera_y)

    # Draw a yellow rectangle as the goal area
    leaf_coords = [coords[i] for i in LEAVES]
    min_x = min(x for x, y in leaf_coords)
    max_x = max(x for x, y in leaf_coords)
    min_y = min(y for x, y in leaf_coords)
    max_y = max(y for x, y in leaf_coords)
    goal.rect = pygame.Rect(min_x - 50, min_y - 50, (max_x - min_x) + 100, (max_y - min_y) + 100)
    pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(goal.rect.left - camera_x, goal.rect.top - camera_y, goal.rect.width, goal.rect.height), border_radius=25)

    # Draw the ship
    srect = ship.rect.move(-camera_x, -camera_y)
    screen.blit(ship.surf, srect.topleft)

    # Display quantum state info
    p_plus, p_minus = qubit.probabilities_in_x_basis()
    screen.blit(font.render("Gate Sequence:", True, (180, 180, 255)), (20, 20))
    screen.blit(font.render(" ".join(qubit.gate_history[-10:]), True, WHITE), (20, 50))
    screen.blit(font.render(f"X P+: {p_plus:.2f}  P-: {p_minus:.2f}", True, (255, 255, 0)), (20, 100))
    screen.blit(font.render("Use H, X, Z, S keys to apply gates", True, (255, 255, 0)), (20, 140))

    # Draw probability bars
    bar_y, bar_width, bar_height = 180, 20, 100
    pygame.draw.rect(screen, GREEN, (35, bar_y + bar_height * (1 - p_plus), bar_width, bar_height * p_plus))
    pygame.draw.rect(screen, RED, (60, bar_y + bar_height * (1 - p_minus), bar_width, bar_height * p_minus))
    screen.blit(font.render("+", True, WHITE), (35, bar_y + bar_height + 5))
    screen.blit(font.render("-", True, WHITE), (62, bar_y + bar_height + 5))

    # Draw Bloch sphere projection
    bx, by, bz = qubit.get_bloch_coordinates()
    center, radius = (105, 410), 60
    pygame.draw.circle(screen, WHITE, center, radius, 1)
    pygame.draw.circle(screen, (0, 255, 255), (int(center[0] + radius * bx), int(center[1] - radius * bz)), 6)
    pygame.draw.line(screen, (100, 100, 100), (center[0] - radius, center[1]), (center[0] + radius, center[1]), 1)
    pygame.draw.line(screen, (100, 100, 100), (center[0], center[1] - radius), (center[0], center[1] + radius), 1)
    screen.blit(font.render("+x", True, WHITE), (center[0] + radius + 5, center[1] - 10))
    screen.blit(font.render("-x", True, WHITE), (center[0] - radius - 37, center[1] - 13))
    screen.blit(font.render("+z", True, WHITE), (center[0] - 10, center[1] - radius - 29))
    screen.blit(font.render("-z", True, WHITE), (center[0] - 10, center[1] + radius))

    # Show full state in ket notation
    screen.blit(font.render(f"|ψ> ~= {qubit.pretty_state()}", True, (200, 200, 255)), (130, 180))

    # End screen
    if game_over:
        message = "You Win!" if game_won else "Game Over!"
        color = (0, 255, 0) if game_won else (255, 0, 0)
        msg_surface = end_font.render(message, True, color)
        screen.blit(msg_surface, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 50))

    pygame.display.flip()
    prev_keys = pressed_keys

pygame.quit()