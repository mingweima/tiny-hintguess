import numpy as np
import torch
from torchvision import datasets, transforms


class MNIST_Encoding:
    def __init__(self):
        self.dataset = datasets.MNIST('../mnist_data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize([int(14), int(14)]),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]))
        self.size = len(self.dataset)
        self.encoding_dim = int(14 * 14)
        self.labels = np.array(self.dataset.targets)

    def __call__(self, pos_array: torch.Tensor, d_model=None):
        pe = torch.zeros((pos_array.shape[0], self.encoding_dim))
        for i, pos in enumerate(pos_array):
            number = pos.item()
            chosen_index = np.random.choice(np.argwhere(self.labels == number).flatten())
            # print(self.dataset[0])
            pe[i, :] = self.dataset[chosen_index][0].flatten()
        return pe


