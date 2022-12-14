from torch.utils.data import Dataset
import utils_for_training as ut


class fr_to_en(Dataset):
    def __init__(self, set_type):
        super().__init__()
        if set_type == "training":
            self.fr_set = ut.open_text_set("data/training/train.fr")
            self.en_set = ut.open_text_set("data/training/train.en")
            print("Dataset is training")
        elif set_type == "validation":
            self.fr_set = ut.open_text_set("data/validation/val.fr")
            self.en_set = ut.open_text_set("data/validation/val.en")
            print("Dataset is validation")
        else:
            raise ValueError('set_type must be "training" or "validation"')

    def __len__(self):
        return len(self.fr_set)

    def __getitem__(self, idx):
        return self.fr_set[idx], self.en_set[idx]
