import torch
from mst.dataloaders.cambridge import CambridgeDataset
from mst.dataloaders.medley import MedleyDBDataset
import pytorch_lightning as pl



class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets = [CambridgeDataset, MedleyDBDataset]):
        self.datasets = datasets


    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
    

    def __len__(self):
        return  len(self.datasets)
    

class CombinedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        length: int,
        num_workers: int = 4,
        batch_size: int = 16,
        train_buffer_size_gb: float = 2.0,
        val_buffer_size_gb: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        concat_dataset = CombinedDataset()
        
    

    # loader = torch.utils.data.DataLoader(
    #     concat_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True
    # )
    #     return loader

    # def val_dataloader(self):
    

    # def test_dataloader(self):
       
if __name__ == "__main__":
    dataset = CombinedDataset()
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, drop_last = True, num_workers=4)

    for i, (track, instrument_id, stereo) in enumerate(dataloader):
       print(i)
       break