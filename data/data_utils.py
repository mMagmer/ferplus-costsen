
import os

import torch
from torch.utils.data import Dataset , DataLoader

import numpy as np 
import albumentations as A
from albumentations.pytorch import ToTensorV2

Spatial_transform = A.OneOf([A.ShiftScaleRotate(p=.5),
                             A.RandomCrop(height=40, width=40,p=.5),
                             A.Perspective(p=.5)],p=.75)

pixel_transform = A.OneOf([A.ColorJitter(p=.5),
                           A.Sharpen(p=.5),
                           A.GaussNoise(),
                           A.Posterize()],p=.75)

transform_train = A.Compose([A.HorizontalFlip(),
                             pixel_transform,
                             Spatial_transform,
                             A.CoarseDropout(max_holes=4,min_holes=1,max_height=4,max_width=4,p=.5,fill_value=128),
                             A.Resize(128,128),
                             A.Normalize(mean=(0.5), std=(0.5),max_pixel_value=255.0),
                             ToTensorV2()
                            ])

transform_weak = A.Compose([A.HorizontalFlip(p=.2),
                            A.ShiftScaleRotate(scale_limit=0.0, rotate_limit=5,p=.1),
                            A.OneOf([A.ColorJitter(),
                                     A.Sharpen(),
                                     A.GaussNoise(),
                                     A.Posterize()],p=.2),
                            A.CoarseDropout(max_holes=4,min_holes=1,max_height=4,max_width=4,fill_value=128,p=.1),
                            A.Resize(128,128),
                            A.Normalize(mean=(0.5), std=(0.5),max_pixel_value=255.0),
                            ToTensorV2()
                           ])

transform_infer = A.Compose([A.Resize(128,128),
                             A.Normalize(mean=(0.5), std=(0.5),max_pixel_value=255.0),
                             ToTensorV2()
                            ])

gmean = lambda p: torch.exp(torch.log(p).mean())

class FERDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_split, transform=None, transform_weak=None):
        """
        Convert a dictionary containing list of images and lables to standard PyTorch definition of Dataset.
        Specifies transforms to apply on images.

        Args:
            data_split: (dict) dictionary containing list of images and labels
            transform: (albumentations) transformation to apply on image
            transform_weak: (albumentations) transformation to apply on image of majority and minority classes with different probability
        """
        self.in_channels=1
        self.num_classes=8
        self.data_split = data_split
        self.transform = transform
        self.transform_weak =  transform_weak
        
        if transform_weak:
            assert transform , 'transfom is "None",you should specify it too if you are using tarnsform_weak'
        
        p = torch.Tensor([36.3419, 26.4458, 12.5597, 12.4088,  8.6819,  0.6808,  2.2951,  0.5860])
        Nprior = (p/gmean(p))
        self.cut_off = torch.sigmoid(-torch.log(Nprior))

    def __len__(self):
        # return size of dataset
        return len(self.data_split['images'])

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = self.data_split['images'][idx]
        label = self.data_split['labels'][idx]
        
        if self.transform_weak:
            t = torch.rand([]).item()
            if t > self.cut_off[label].item():
                augmented = self.transform_weak(image=image)
            
            else:
                augmented = self.transform(image=image)
            
            image = augmented['image']
                
        elif self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, self.data_split['labels'][idx]

def fetch_data(data_path="./data/fer2013/fer2013.csv",new_label_path="./data/fer2013/fer2013new.csv"):
    """
    Fetches the Data images and label from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        new_label_path: (string) directory containing fer+ dataset label
    Returns:
        data: (dict) contains the DataSets object for each fold ['train', 'val', 'test']
    """
    new_classes=("neutral","happiness","surprise","sadness","anger","disgust","fear","contempt","unknown","NF")
    
    with open(data_path) as f: # read all the csv using readlines
        data = f.readlines()
        
    votes = np.genfromtxt(new_label_path,delimiter=',',skip_header=1,usecols=list(range(2,12)))
    majority = votes[:,:8].argmax(axis=1)
    
    valid_idx = np.logical_not(((votes[:,9]>5)+(votes[:,8]>4))) # remove unknown and NF data
    
     
    splits_keys = {'Training' : 'train', 'PublicTest' : 'val', 'PrivateTest': 'test'}
    
    splits = {'train': {'images': [], 'labels': [] }, 'val': {'images': [], 'labels': [] },
                'test': {'images': [], 'labels': [] } }
    
    for idx in np.argwhere(valid_idx)[:,0]:
    
        _, img, usage = data[idx+1].split(",") #  +1 to skip first row (column name)
        
        img = img.split(" ") # because the pixels are seperated by space
        img = np.array(img, 'uint8') # just make sure it is int not str
        img = img.reshape(48,48,1) # change shape from 2304 to 48 * 48
        
        label = majority[idx]
        
        splits[splits_keys[usage.strip()]]['images'].append(img)
        splits[splits_keys[usage.strip()]]['labels'].append(label)
        
    return splits , new_classes[:8]

if __name__ == '__main__':
    # this script plot some example of FER+ dataset.
    
    data_splits ,classes = fetch_data()
    trainset = FERDataset(data_splits['train'])
    
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(12, 5),
                            subplot_kw={'xticks': [], 'yticks': []})

    images, labels = trainset[0:12]
    for ax, image, label in zip(axs.flat, images, labels):
        ax.imshow(image, cmap='gray',interpolation='bilinear', vmin=0, vmax=255)
        ax.set_title(classes[label])

    plt.tight_layout()
    plt.show()