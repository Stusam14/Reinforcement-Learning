import random
import numpy as np
from FourRooms import FourRooms

class Courier:
    def __init__(self, room_env: FourRooms):
        # Initialize Courier agent with environment and parameters
        self.visited_states = None
        self.q_matrix = None
        self.room = room_env
        self.reward_matrix = None
        self.package_positions = np.zeros((3, 1))
        self.room_size = 12
        self.width = 4
        self.height = 144
        self.num_packages = 3
        self.actions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        self.episodes = 10000
        self.epsilon = 1
        self.start_decay = 1
        self.end_decay = self.episodes // 2
        self.decay_value = self.epsilon / (self.end_decay - self.start_decay)
        self.show_interval = 20
        self.prev_grid_cell = []
        self.prev_remaining_packages = []

    def available_actions(self, current_package: int, state: int) -> [int]:
        # Return available actions for given state and package
        return [idx for idx, element in enumerate(self.reward_matrix[current_package, state]) if element != -1]

    def next_state(self, current_state: int, action: int) -> int:
        # Return next state given current state and action
        next_pos = np.array([current_state % self.room_size, current_state // self.room_size]) + self.actions[action]
        return next_pos[1] * self.room_size + next_pos[0]

    def calculate_reward(self) -> np.array:
        # Calculate reward matrix for the environment
        reward_matrix = np.zeros((self.num_packages, self.height, self.width), dtype=int)
        for package in range(self.num_packages):
            for state in range(self.height):
                for idx, action in enumerate(self.actions):
                    next_pos = np.array([state % self.room_size, state // self.room_size]) + action
                    if not (0 <= next_pos[0] < self.room_size and 0 <= next_pos[1] < self.room_size):
                        reward_matrix[package, state, idx] = -1
        return reward_matrix

    def calculate_action_reward(self, current_state: int, next_state: int, action: int, package: int,
                                grid_cell: int, is_terminal: bool) -> float:
        # Calculate reward for a given action
        if current_state == next_state:
            self.reward_matrix[package, current_state, action] = -1
            return 0.0
        else:
            return grid_cell

    def q_exploration(self, episodes: int, gamma: float = 0.9) -> None:
        # Exploration phase of Q-learning
        q_table = self.q_matrix.copy()
        visit_table = self.visited_states.copy()
        x, y = self.room.getPosition()
        state = y * self.room_size + x
        goal_reached = False
        remaining_packages = self.room.getPackagesRemaining()
        current_package = {3: 0, 2: 1, 1: 2}.get(remaining_packages, 0)
        collected_packages = None
        next_remaining_packages = 0

        while not goal_reached:
            state_actions = self.available_actions(current_package, state)
            action = np.argmax(q_table[current_package, state]) if random.random() > self.epsilon else random.choice(state_actions)
            grid_cell, current_pos, remaining_packages, is_terminal = self.room.takeAction(action)
            next_state = current_pos[1] * self.room_size + current_pos[0]

            if episodes > 1:
                if self.prev_grid_cell and self.prev_grid_cell[0] != grid_cell and collected_packages is None and remaining_packages < 3:
                    break
                if self.prev_grid_cell and self.prev_grid_cell[1] != grid_cell and collected_packages == 1 and grid_cell > 0:
                    break

            reward = self.calculate_action_reward(state, next_state, action, current_package, grid_cell, is_terminal)

            visit_table[current_package, state, action] += 1
            learning_rate = 1 / visit_table[current_package, state, action]

            q_table[current_package, state, action] += learning_rate * (
                    reward + gamma * (
                    np.max(q_table[current_package, next_state]) - q_table[current_package, state, action]
            ))
            next_remaining_packages = {2: 0, 1: 1, 0: 2}.get(remaining_packages, 0)
            if remaining_packages == 2 and collected_packages is None:
                q_table, visit_table, collected_packages, current_package = self.store_move_info(q_table, visit_table, grid_cell, remaining_packages, 1, next_remaining_packages)

            if remaining_packages == 1 and collected_packages == 1:
                q_table, visit_table, collected_packages, current_package = self.store_move_info(q_table, visit_table, grid_cell, remaining_packages, 2, next_remaining_packages)

            if remaining_packages == 0 and collected_packages == 2:
                q_table, visit_table, collected_packages, current_package = self.store_move_info(q_table, visit_table, grid_cell, remaining_packages, 3, next_remaining_packages)

            goal_reached = is_terminal
            state = next_state

        if self.end_decay >= episodes >= self.start_decay:
            self.epsilon -= self.decay_value

    def store_move_info(self, q_table, visit_table, grid_cell, remaining_packages, collected_packages,
                        next_packages) -> (np.array, np.array, int, int):
        # Store information about moves
        self.update_q_table(q_table, visit_table)
        self.prev_grid_cell.append(grid_cell)
        self.prev_remaining_packages.append(remaining_packages)
        q_table = self.q_matrix.copy()
        visit_table = self.visited_states.copy()
        return q_table, visit_table, collected_packages, next_packages + 1

    def update_q_table(self, q_table, visit_table) -> None:
        # Update Q-table and visit table
        self.q_matrix = q_table.copy()
        self.visited_states = visit_table.copy()

    def q_exploit(self) -> None:
        # Exploitation phase of Q-learning
        x, y = self.room.getPosition()
        goal_reached = False
        state = self.room_size * y + x
        remaining_packages = self.room.getPackagesRemaining()
        current_package = {3: 0, 2: 1, 1: 2}.get(remaining_packages, 0)
        path = []

        while not goal_reached:
            path.append(state)
            action = np.argmax(self.q_matrix[current_package, state])
            reward, current_pos, remaining_packages, is_terminal = self.room.takeAction(action)

            goal_reached = is_terminal
            if goal_reached:
                break

            next_package = {3: 0, 2: 1, 1: 2}.get(remaining_packages, 0)
            next_state = current_pos[1] * self.room_size + current_pos[0]
            state = next_state
            current_package = next_package

    def q_learning(self) -> np.array:
        # Q-learning process
        self.q_matrix = np.zeros((self.num_packages, self.height, self.width), dtype=float)
        self.visited_states = np.zeros((self.num_packages, self.height, self.width), dtype=int)
        self.reward_matrix = self.calculate_reward()

        best_learning = None
        early_stop = 144 * 4
        best_epoch = None

        for episode in range(1, self.episodes):
            self.q_exploration(episode)
            avg_state_visits = np.mean(self.q_matrix)
            if best_learning is None or avg_state_visits > best_learning:
                best_learning = avg_state_visits
                best_epoch = episode
            if best_epoch + early_stop <= episode:
                break
            self.room.newEpoch()

        return self.visited_states

    def evaluate_agent(self) -> None:
        # Evaluate agent's performance
        self.room.newEpoch()
        self.q_exploit()

def main():
    # Initialize environment and agent
    room = FourRooms("multi")
    courier = Courier(room)
    print("Training starting...")
    # Train the agent
    courier.q_learning()
    # Evaluate the agent
    courier.evaluate_agent()
    room.showPath(-1)
    print("Done!")

if __name__ == "__main__":
    main()
