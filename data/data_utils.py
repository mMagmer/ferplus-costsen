import numpy as np 
import os


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
        img = np.array(img, 'int') # just make sure it is int not str
        img = img.reshape(48,48) # change shape from 2304 to 48 * 48
        
        label = majority[idx]
        
        splits[splits_keys[usage.strip()]]['images'].append(img)
        splits[splits_keys[usage.strip()]]['labels'].append(label)
        
    return splits , new_classes[:8]

if __name__ == '__main__':
    s , classes = fetch_data()
    print(np.stack(s['test']['images']).shape)