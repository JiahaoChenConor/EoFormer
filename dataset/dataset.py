import os
import json
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from monai.data import CacheDataset
from monai.transforms import Resize, MapTransform

def split_data(datalist, basedir, with_infer=False):
    with open(datalist) as f:
        json_data = json.load(f)

    def process_split(split):
        processed = []
        for d in split:
            for k, v in d.items():
                if isinstance(v, list):
                    d[k] = [os.path.join(basedir, iv) for iv in v]
                elif isinstance(v, str):
                    d[k] = os.path.join(basedir, v) if len(v) > 0 else v
            processed.append(d)
        return processed

    train = process_split(json_data.get('train', []))
    valid = process_split(json_data.get('valid', []))
    test  = process_split(json_data.get('test', []))

    if with_infer and 'infer' in json_data:
        inference = process_split(json_data['infer'])
        return train, valid, test, inference
    else:
        return train, valid, test


def get_inference_dataset_brats(data_path: str, json_file: str, transform=None):
    _, _, _, inference_list = split_data(json_file, data_path, with_infer=True)
    inference_set = CacheDataset(data=inference_list, cache_rate=0.0, transform=transform['inference'])
    return inference_set, inference_list

def get_dataset_brats(data_path: str, 
                  json_file: str, 
                  transform=None,
                 )-> Tuple[Dataset, Dataset]:
    
    train_list, valid_list, test_list = split_data(json_file, data_path)
    
    train_set = CacheDataset(
                            data = train_list,
                            cache_rate=0.0,
                            transform = transform['train']
                            )
    valid_set = CacheDataset(
                            data = valid_list,
                            cache_rate=0.0,
                            transform = transform['valid']
                            )
    test_set = CacheDataset(
                            data = test_list,
                            cache_rate=0.0,
                            transform = transform['valid']
                            )
    
    return train_set, valid_set, test_set, train_list, valid_list, test_list