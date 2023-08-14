import torch
from mst.dataloaders.cambridge import CambridgeDataset
from mst.dataloaders.medley import MedleyDBDataset
import pytorch_lightning as pl



class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        # print(i)
        for d in self.datasets:
            tracks, id, stereo  = d[i]
            # print(tracks.shape)
            # print(id)
            # print(stereo)

        return tracks, id, stereo
        #return self.data[i]

    def __len__(self):
        #min(len(d) for d in self.datasets)
        return  min(len(d) for d in self.datasets)
    

class CombinedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        medley_rootdir: list = ["/import/c4dm-datasets/MedleyDB_V1/V1", "/import/c4dm-datasets/MedleyDB_V2/V2"],
        cambridge_rootdir: list = ["/import/c4dm-multitrack-private/C4DM Multitrack Collection/mixing-secrets"],
        length: int = 524288,
        num_workers: int = 4,
        batch_size: int = 16,
        min_tracks: int = 4,
        max_tracks: int = 20,
        train_buffer_size_gb: float = 2.0,
        val_buffer_size_gb: float = 0.1,
        num_examples_per_epoch: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.current_epoch = -1
        self.max_tracks = 1
        
        # self.num_workers = num_workers
        # self.batch_size = batch_size
        # self.dataset_1 = dataset_1
        # self.dataset_2 = dataset_2
        
    
    def setup(self, stage=None):
        pass

    

    def train_dataloader(self):
        self.current_epoch += 1

        # set the max number of tracks (increase every 10 epochs)
        self.max_tracks = (self.current_epoch // 10) + 1

        if self.max_tracks > self.hparams.max_tracks:
            self.max_tracks = self.hparams.max_tracks

        # batch_size = (self.hparams.max_tracks // self.max_tracks) // 2
        # if batch_size < 1:
        #    batch_size = 1

        
        print(f"Current epoch: {self.current_epoch} with max_tracks: {self.max_tracks}")

        # dataset =CombinedDataset(CambridgeDataset(root_dirs=["/import/c4dm-multitrack-private/C4DM Multitrack Collection/mixing-secrets"]),
        #                      MedleyDBDataset(["/import/c4dm-datasets/MedleyDB_V1/V1", "/import/c4dm-datasets/MedleyDB_V2/V2"]))
        #print(len(self.train_dataset))
        self.train_dataset =CombinedDataset(CambridgeDataset(root_dirs=self.hparams.cambridge_rootdir,
                                                        subset = "train",
                                                        min_tracks = self.hparams.min_tracks,
                                                        max_tracks = self.hparams.max_tracks,
                                                        length = self.hparams.length,
                                                        buffer_size_gb=self.hparams.train_buffer_size_gb,
                                                        num_examples_per_epoch=self.hparams.num_examples_per_epoch,
                                                        ), MedleyDBDataset(root_dirs=self.hparams.medley_rootdir,
                                                        subset = "train",
                                                        min_tracks = self.hparams.min_tracks, 
                                                        max_tracks = self.hparams.max_tracks, 
                                                        length = self.hparams.length, 
                                                        buffer_size_gb=self.hparams.train_buffer_size_gb,
                                                        num_examples_per_epoch=self.hparams.num_examples_per_epoch,
                                                        ))
        
        loader = torch.utils.data.DataLoader(
                                            self.train_dataset,
                                            batch_size=self.hparams.batch_size, 
                                            shuffle=True, 
                                            drop_last = True, 
                                            num_workers=self.hparams.num_workers)
        return loader
    
    def val_dataloader(self):
        self.val_dataset =CombinedDataset (CambridgeDataset(root_dirs=self.hparams.cambridge_rootdir,
                                                        subset = "val",
                                                        min_tracks = self.hparams.min_tracks,
                                                        max_tracks = self.hparams.max_tracks,
                                                        length = self.hparams.length,
                                                        buffer_size_gb=self.hparams.val_buffer_size_gb,
                                                        num_examples_per_epoch=self.hparams.num_examples_per_epoch,
                                                        ), MedleyDBDataset(root_dirs=self.hparams.medley_rootdir,
                                                        subset = "val",
                                                        min_tracks = self.hparams.min_tracks, 
                                                        max_tracks = self.hparams.max_tracks, 
                                                        length = self.hparams.length, 
                                                        buffer_size_gb=self.hparams.val_buffer_size_gb,
                                                        num_examples_per_epoch=self.hparams.num_examples_per_epoch,
                                                        ))
        loader = torch.utils.data.DataLoader(self.val_dataset,
                                             batch_size=self.hparams.batch_size, 
                                             shuffle=False, 
                                             drop_last = True, 
                                             num_workers=1)
        return loader
    
    def test_dataloader(self):
        self.test_dataset =CombinedDataset (CambridgeDataset(root_dirs= self.hparams.cambridge_rootdir, 
                                                        subset = "test",
                                                        min_tracks = self.hparams.min_tracks,
                                                        max_tracks = self.hparams.max_tracks,
                                                        length = self.hparams.length,
                                                        buffer_size_gb=self.hparams.val_buffer_size_gb,
                                                        num_examples_per_epoch=self.hparams.num_examples_per_epoch,
                                                        ), MedleyDBDataset(root_dirs=self.hparams.medley_rootdir,
                                                        subset = "test",
                                                        min_tracks = self.hparams.min_tracks, 
                                                        max_tracks = self.hparams.max_tracks, 
                                                        length = self.hparams.length, 
                                                        buffer_size_gb=self.hparams.val_buffer_size_gb,
                                                        num_examples_per_epoch=self.hparams.num_examples_per_epoch,
                                                        ))
        loader = torch.utils.data.DataLoader(self.test_dataset,
                                             batch_size=1, 
                                             shuffle=False, 
                                             drop_last = False, 
                                             num_workers=1)
        return loader
    
        
    # if __name__ == "__main__":
        
    #     train_dataset =CombinedDataset(CambridgeDataset(root_dirs=["/data/scratch/acw639/Cambridge/mixing-secrets"]),
    #                                     MedleyDBDataset(["/data/scratch/acw639/Medley/V1", "/data/scratch/acw639/Medley/MedleyDB_V2/V2"]))
    #     print(len(train_dataset))
    #     loader = torch.utils.data.DataLoader(train_dataset,batch_size=4, shuffle=True, drop_last = True, num_workers=0)
    #     print("dataloader")
    #     print(len(loader))
    #     for i, (tracks, id, stereo) in enumerate(loader):
    #         print(i)
    #         print(tracks.shape)
    #         print(id.shape)
    #         print(stereo.shape)
    #         break
        
            