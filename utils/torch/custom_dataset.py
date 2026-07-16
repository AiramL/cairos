import torch

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):

        self.x = torch.from_numpy(x).float().permute(0, 3, 1, 2)
        self.y = torch.from_numpy(y).long()

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        return self.x[index], self.y[index]
