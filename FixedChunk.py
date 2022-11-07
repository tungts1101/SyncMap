from typing import Tuple
import torch
import numpy as np

class FixedChunk:
    def __init__(self, num_states_per_chunk: list, time_delay: int, num_remembered: int):
        """A problem where chunks are a mutually disjoint sets of states.

        Args:
            num_states_per_chunk (list): A list of number of states for each chunk.
            time_delay: A number of steps before state transition happens.
            num_remembered: A number of states that will be remembered
        """        
        self.time_delay   = time_delay
        self.time_counter = 1

        self.total_num_chunks = len(num_states_per_chunk)
        self.total_num_states = sum(num_states_per_chunk)
        states = np.eye(self.total_num_states)
        self.states = []
        current_index = 0
        for num_state in num_states_per_chunk:
            chunk = []
            for _ in range(num_state):
                chunk.append(states[current_index])
                current_index += 1
            self.states.append(chunk)

        self.chunk_index = np.random.randint(self.total_num_chunks)
        self.current_index = 0
        self.num_rembered  = num_remembered
        self.remembered_states = []
        self.true_labels = [i for i,j in enumerate(num_states_per_chunk) for _ in range(j)]
        self.state = None

    def update(self):
        """Update state by time counter changes.
        """
        self.time_counter += 1
        if self.time_counter >= self.time_delay:
            self.time_counter = 1
            self.current_index += 1
            if self.state.any() != None: self.remembered_states.append(self.state)
            if len(self.remembered_states) > self.num_rembered:
                self.remembered_states = self.remembered_states[1:]

            if self.current_index >= len(self.states[self.chunk_index]):
                self.chunk_index = np.random.randint(self.total_num_chunks)
                self.current_index = 0

    def get_input(self):
        """Get input value in each step.

        Returns:
            An input value - an exponentially decaying vector with the same size 
            as the number of states.
        """
        self.state = self.states[self.chunk_index][self.current_index]
        input = self.state * np.exp(-0.1 * self.time_counter)
        for time_frame, old_state in enumerate(self.remembered_states):
            input += old_state * np.exp(-0.1 * (self.time_delay * (len(self.remembered_states) - time_frame)) + self.time_counter)
        return input

    def get_sequence(self, sequence_size: int) -> Tuple:
        """Get sequence of input values.

        Args:
            sequence_size (int): A size of sequence.
        
        Returns:
            A sequence of input values.
        """
        sequence = []
        for _ in range(sequence_size):
            sequence.append(self.get_input())
            self.update()
        return torch.tensor(np.array(sequence))
    
    def get_true_labels(self) -> list:
        """Get chunk label for each state.

        Returns:
            A list of chunk labels.
        """ 
        return self.true_labels