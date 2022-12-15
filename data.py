from torch.utils.data import Dataset
import utils


class fr_to_en(Dataset):
    """
    pytorch dataloader 사용을 위한 class
    """

    def __init__(self, set_type):
        super().__init__()
        if set_type == "training":
            self.src_lang = utils.open_text_set("data/training/train.fr")
            self.trg_lang = utils.open_text_set("data/training/train.en")

            print('► Dataset is "training"')

        elif set_type == "validation":
            self.src_lang = utils.open_text_set("data/validation/val.fr")
            self.trg_lang = utils.open_text_set("data/validation/val.en")

            print('► Dataset is "validation"')

        else:
            raise ValueError('set_type must be "training" or "validation"')

    def __len__(self):
        return len(self.src_lang)

    def __getitem__(self, idx):
        return self.src_lang[idx], self.trg_lang[idx]
