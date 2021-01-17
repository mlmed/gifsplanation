

import torchvision, torchvision.transforms

import sys, os
sys.path.insert(0,"../torchxrayvision/")
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import glob
import numpy as np
import skimage, skimage.filters
import captum, captum.attr
import torch, torch.nn
import pickle
import attribution
import pandas as pd



def get_data(dataset_str, transforms=True):
    
    dataset_dir = "/home/groups/akshaysc/joecohen/"
    
    if transforms:
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    else:
        transform = None
    
    
    datasets = []
    
    if "covid" in dataset_str:
        dataset = xrv.datasets.COVID19_Dataset(
            imgpath=dataset_dir + "/covid-chestxray-dataset/images",
            csvpath=dataset_dir + "/covid-chestxray-dataset/metadata.csv",
            transform=transform)
        datasets.append(dataset)

    if "pc" in dataset_str:
        dataset = xrv.datasets.PC_Dataset(
            imgpath=dataset_dir + "/PC/images-224",
            transform=transform, unique_patients=False)
        datasets.append(dataset)

    if "rsna" in dataset_str:
        dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
            imgpath=dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
            transform=transform,unique_patients=False, pathology_masks=True)
        datasets.append(dataset)
        
    if "nih" in dataset_str:
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=dataset_dir + "/NIH/images-224", 
            transform=transform, unique_patients=False, pathology_masks=True)
        datasets.append(dataset)
        
    if "siim" in dataset_str: 
        dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(
            imgpath=dataset_dir + "SIIM_TRAIN_TEST/dicom-images-train/",
            csvpath=dataset_dir + "SIIM_TRAIN_TEST/train-rle.csv",
            transform=transform, unique_patients=False, masks=True)
        datasets.append(dataset)
        
    if "chex" in dataset_str:
        dataset = xrv.datasets.CheX_Dataset(
            imgpath=dataset_dir + "/CheXpert-v1.0-small",
            csvpath=dataset_dir + "/CheXpert-v1.0-small/train.csv",
            transform=transform, unique_patients=False)
        datasets.append(dataset)
        
    if "google" in dataset_str:
        dataset = xrv.datasets.NIH_Google_Dataset(
            imgpath=dataset_dir + "/NIH/images-224",
            transform=transform)
        datasets.append(dataset)
        
    if "mimic_ch" in dataset_str:
        dataset = xrv.datasets.MIMIC_Dataset(
            imgpath=dataset_dir + "/images-224-MIMIC/files",
            csvpath=dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
            metacsvpath=dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
            transform=transform, unique_patients=False)
        datasets.append(dataset)
        
        



    newlabels = set()
    for d in datasets:
        newlabels = newlabels.union(d.pathologies)

    newlabels = sorted(newlabels)
    #newlabels.remove("Support Devices")
    #print("labels",list(newlabels))
    for d in datasets:
        xrv.datasets.relabel_dataset(list(newlabels), d, silent=True)
    
    if len(datasets) > 1:
        dmerge = xrv.datasets.Merge_Dataset(datasets)
    else:
        dmerge = datasets[0]
    
    print(dmerge.string())
    
    return dmerge
    
    
    
    
    
    
    
    
        