import numpy as np


class TinyHintGuessGame:
    def __init__(self, ndim: int = 10, hsize: int = 4) -> None:
        # ndim: the numbers in the game for hints and guesses; default: 1-10
        self.ndim = ndim
        self.hsize = hsize
        self.nums_array = np.array(list(range(ndim)))
        self.h1, self.h2, self.h1_per, self.h2_per, self.target = None, None, None, None, None
        self.hint, self.guess, self.reward = None, None, None
        self.steps = 0

    def reset(self, initial_config: dict = None) -> np.ndarray:
        self.h1, self.h2, self.h1_per, self.h2_per, self.target = None, None, None, None, None
        self.hint, self.guess, self.reward = None, None, None
        self.steps = 0
        if initial_config is None:
            self.h1 = np.random.choice(self.nums_array, self.hsize)
            self.h2 = np.random.choice(self.nums_array, self.hsize)
            self.target = np.random.choice(self.h2, 1)[0]
        else:
            self.h1 = initial_config['h1']
            self.h2 = initial_config['h2']
            self.target = initial_config['target']
        self.h1_per = np.random.permutation(self.h1)  # permute hands
        self.h2_per = np.random.permutation(self.h2)
        return np.hstack((self.h2, self.h1, np.array(
            self.target)))  # this is obs to hinter; note that the action space (here h1) must be right before the target which is at the end

    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        if self.steps == 0:
            self.hint = self.h1[action]
            self.steps += 1
            return np.hstack((self.h1_per, self.h2, np.array(self.hint))), 0, False, {}
        if self.steps == 1:
            self.guess = self.h2[action]
            if self.guess == self.target:
                self.reward = 10.
            else:
                self.reward = -10.
            return None, self.reward, True, {}
