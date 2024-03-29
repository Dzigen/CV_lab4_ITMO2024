import pandas as pd
import torch
from PIL import Image
from datasets import Dataset
from dataclasses import dataclass
import numpy as np
from src.preprocess import generate_noise

@dataclass
class LearningConfig:
    gen_lr: float
    gen_betas: list
    dis_lr: float
    dis_betas: list
    epochs: int
    batch: int
    device: str
    run_name: str
    model_type: str
    inceptionv3_batch: int

#
class CustomCelebDataset(Dataset):
    def __init__(self, data_table_path, processor, data_part, base_dir='.', omit_label=False):
        data = pd.read_csv(data_table_path, sep=';')

        self._data = data[data['part'] == data_part].reset_index(drop=True)

        self.labels_map = {label: i for i, label in enumerate(data['label'].unique())}
        self.base_dir = base_dir
        self.processor = processor
        self.omit_label = omit_label

    def __len__(self):
        return self._data.shape[0]
    
    def __getitem__(self, idx):
        image_path = f"{self.base_dir}/{self._data['croped_path'][idx]}/{self._data['images_name'][idx]}"
        
        image = Image.open(image_path).convert('RGB')
        
        image_tensor = self.processor(image) if self.processor is not None else np.array(image)
        image.close()
        
        if self.omit_label:
            return image_tensor
        else:
            label = self.labels_map[self._data['label'][idx]]
            return image_tensor, label
    
    def __getitems__(self, idxs):
        return [self.__getitem__(idx) for idx in idxs]

def custom_collate(data):

    images = torch.cat([torch.unsqueeze(item[0], 0) for item in data], 0)
    labels = torch.tensor([item[1] for item in data])

    return {
        "images": images, 
        "labels": labels
    }

#
class CustomFakeDataset(torch.utils.data.Dataset):
    def __init__(self, gen_model, real_dataset_size, preprocessor, device):
        self.dataset_size = real_dataset_size
        self.gen_model = gen_model
        self.preprocessor = preprocessor
        self.device = device

    def __getitem__(self, index):

        z = generate_noise(1).to(self.device)
        y = torch.randint(0,2,size=(1, 1)).to(self.device)

        out = self.gen_model(z, y).detach().cpu()

        pre_out = self.preprocessor(out[0]) if self.preprocessor is not None else out
        return pre_out

    def __len__(self):
        return self.dataset_size
    
    def __getitems__(self, idxs):
        return [self.__getitem__(idx) for idx in idxs]