import numpy as np
from FourRooms import FourRooms

# Constants
NUM_STATES = 144
NUM_ACTIONS = 4
GAMMA = 0.8
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 200
EPOCHS = 500

# Initialize Q-table and Rewards
Q_table = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=float)
Rewards = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=int)

# Initialize FourRooms Environment
fourRoomsObj = FourRooms('simple')

# Directions: UP, DOWN, LEFT, RIGHT
directions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

def populate_rewards():
    """
    Populate the Rewards table.
    """
    for state in range(NUM_STATES):
        x, y = state % 12, state // 12
        for i, (dx, dy) in enumerate(directions):
            next_x, next_y = x + dx, y + dy
            if 0 <= next_x < 12 and 0 <= next_y < 12:
                Rewards[state, i] = 0
            else:
                Rewards[state, i] = -1

def take_action(state, action):
    """
    Take an action and return the new state.
    """
    x, y = state % 12, state // 12
    new_x, new_y = x + directions[action][0], y + directions[action][1]
    return new_y * 12 + new_x

def learning_q(epoch):
    """
    Q-learning algorithm.
    """
    global EPSILON_START
    EPSILON_START -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS

    state = fourRoomsObj.getPosition()[1] * 12 + fourRoomsObj.getPosition()[0]
    done = False

    while not done:
        # Exploration vs. Exploitation
        if np.random.rand() < EPSILON_START:
            action = np.random.choice(NUM_ACTIONS)
        else:
            action = np.argmax(Q_table[state])

        # Take action
        grid_cell, current_pos, _, is_terminal = fourRoomsObj.takeAction(action)
        next_state = current_pos[1] * 12 + current_pos[0]

        # Update Q-table
        old_value = Q_table[state, action]
        next_max = np.max(Q_table[next_state])
        new_value = (1 - 1 / (1 + epoch)) * old_value + 1 / (1 + epoch) * (grid_cell + GAMMA * next_max)
        Q_table[state, action] = new_value

        # Move to the next state
        done = is_terminal
        state = next_state

def exploit():
    """
    Exploit the learned policy.
    """
    state = fourRoomsObj.getPosition()[1] * 12 + fourRoomsObj.getPosition()[0]
    done = False

    while not done:
        action = np.argmax(Q_table[state])
        _, current_pos, _, is_terminal = fourRoomsObj.takeAction(action)
        state = current_pos[1] * 12 + current_pos[0]
        done = is_terminal

    fourRoomsObj.showPath(-1)

def main():
    # Populate rewards and run Q-learning
    populate_rewards()
    for epoch in range(1, EPOCHS + 1):
        fourRoomsObj.newEpoch()
        learning_q(epoch)
    
    # Exploit the learned policy
    fourRoomsObj.newEpoch()
    exploit()

if __name__ == "__main__":
    main()
