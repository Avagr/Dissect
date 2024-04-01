import abc

from torch.utils.data import Dataset


class BaseDataset(Dataset, abc.ABC):
    wandb_columns = None

    @abc.abstractmethod
    def wandb_repr(self, idx, prediction):
        raise NotImplementedError
