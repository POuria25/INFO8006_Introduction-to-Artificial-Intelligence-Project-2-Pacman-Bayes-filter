import math
import numpy as np

from pacman_module.game import Agent, Directions, manhattanDistance


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def calculate_neighbor_probabilities(self, walls, i, j, position, fear_factor):
        """
        Calculate the probabilities of moving to neighboring cells based on the 
        current position and fear factor.

        Arguments:
            walls: The W x H grid of walls.
            i: The current x-coordinate.
            j: The current y-coordinate.
            position: The current position of Pacman.
            fear_factor: The fear level of the ghost.

        Returns:
            A dictionary with keys as neighboring cell coordinates and values as 
            the corresponding probabilities.
        """ 
        
        probabilities = {}
        previous_distance = manhattanDistance(position, (i, j))
        
        for dx, dy, in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = i + dx, j +dy
            if 0 <= x < walls.width and 0 <= y < walls.height and not walls[x][y]:
                dist = manhattanDistance(position, (x, y))
                probability = 2**fear_factor if dist >= previous_distance else 1
                probabilities[(x, y)] = probability
                
        return probabilities
    
    def normalize_distribution(distribution):
        """
        Normalize the given probability distribution.

        Arguments:
            distribution: A dictionary where keys are events and values are their 
                          corresponding probabilities.

        Returns:
            The normalized distribution where the sum of all probabilities equals 1.
        """
        
        total = sum(distribution.values())
        if total > 0:
            for key in distribution:
                distribution[key] /= total
        return distribution
                
    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """
        
        T = np.zeros((walls.width, walls.height, walls.width, walls.height))
        fear_factor = {'fearless': 0, 'afraid': 1, 'terrified': 3}[self.ghost]
        
        for i in range(walls.width):
            for j in range(walls.height):
                if not walls[i][j]:  # If it's not a wall
                    # Get unnormalized probabilities for neighboring cells
                    neighbor_probs = self.calculate_neighbor_probabilities(walls, i, j, position, fear_factor)
                    # Normalize the probabilities
                    normalized_probs = self.normalize_distribution(neighbor_probs)
                    
                    # Assign the normalized probabilities to the transition matrix
                    for (x, y), prob in normalized_probs.items():
                        T[i, j, x, y] = prob
        
        return T

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """
        observation_matrix = np.zeros((walls.width, walls.height))
        n, p = 4, 1/2
        
        
        for i in range(walls.width):
            for j in range(walls.height):
                if not walls[i][j]:
                    # Distance between the current position and the coordinates (i, j)
                    distance = manhattanDistance(position, (i, j))
                    # Adjusted distance based on the evidence, distance, and parameters n and p
                    adjusted_distance = evidence - distance + n * p
                    # If the adjusted distance is negative, set the observation_matrix entry to 0
                    if adjusted_distance < 0:
                        observation_matrix[i, j] = 0
                    else:
                        # Otherwise, calculate the binomial probability and update the observation matrix
                        observation_matrix[i, j] = math.comb(n, int(adjusted_distance)) * (p**adjusted_distance) * ((1 - p)**(n-adjusted_distance))
        
        return observation_matrix

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """

        T = self.transition_matrix(walls, position)
        O = self.observation_matrix(walls, evidence, position)
        
        predicted_belief = np.tensordot(belief, T, axes=([0,1], [0,1]))
        updated_belief = np.multiply(predicted_belief, O)
        update = updated_belief / np.sum(updated_belief)
        
        return update

        

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        
        # A COMPLETER
        
        ghost_positions = [np.unravel_index(np.argmax(belief), belief.shape) for belief in beliefs]
        target = min(ghost_positions, key=lambda ghost: manhattanDistance(position, ghost))
        return a_star(position, target, walls)

        return Directions.STOP

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
