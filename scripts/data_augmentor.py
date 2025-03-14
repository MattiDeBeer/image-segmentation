#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:29:39 2025

@author: matti
"""
from pathlib import Path
from PIL import Image
import random
import torch
from torchvision import transforms
from tqdm import tqdm
import os
import h5py

import numpy as np
import matplotlib.pyplot as plt

class CustomTransform:
    """
    Defines a custom transform class
    This ensures that the same transform is applied to both the image and label
    """
    def __init__(self, image_transform, label_transform):
        """
        Parameters
        ----------
        image_transform : torchvision.transforms object
            The transform you whish to apply to the image.
        label_transform : torchvision.transforms object
            The transform you whish to apply to the label.

        Returns
        -------
        None.

        """
        #sets transforms as an attribute of the calss
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __call__(self, image, label):
        """
        Parameters
        ----------
        image : PIL image object
            The image to be transformed
        label : PIL image object
            The label to be transformed

        Returns
        -------
        image : PIL image object
            The transformed image
        label : PIL image object
            The transformed label

        """
        #Generate a random seed
        seed = torch.randint(0, 2**32, (1,)).item()

        #set set torch random state to seed
        random.seed(seed)
        torch.manual_seed(seed)

        #generate transformed image
        image = self.image_transform(image)

        #set set torch random state to the previous seed
        random.seed(seed)
        torch.manual_seed(seed)

        #generate transformed label
        label = self.label_transform(label)

        #return labels and images
        return image, label

def fetch_dataset_filenames(image_directory = '../Dataset/TrainVal/color/', label_directory = '../Dataset/TrainVal/label/'):
    """
    Parameters
    ----------
    image_directory : string, optional
        The directory of the images. The default is '../Dataset/TrainVal/color/'.
    label_directory : string, optional
        The directory of the labels. The default is '../Dataset/TrainVal/label/'.

    Returns
    -------
    filenames: list
        A list of the commom filenames found in both directories

    """

    #define image and label paths
    data_path = Path(image_directory)
    label_path = Path(label_directory)

    #find all image files and remoce the filetype
    data_filenames = set([f.name.replace('.jpg','') for f in list(data_path.glob("*.jpg"))])
    label_filenames = set([f.name.replace('.png','') for f in list(label_path.glob("*.png"))])

    #Find the filename intersectopn
    common_files = data_filenames & label_filenames

    #using the common files, generate the full paths
    data_files = [image_directory + file + '.jpg' for file in common_files]
    label_files = [label_directory + file + '.png' for file in common_files]

    return list(data_files), list(label_files), list(common_files)

def transform_datapoint(image,label,identity_map=False,image_dim = 300):
    """
    Parameters
    ----------
    image : PIL Image object
        The image to be transformed
    label : PIL Image object
        The label to be transformed
    identity_map : bool, optional
        This flag allows one to only crop the image, but not rotate, scale or change it's colours. 
        The default is False.

    Returns
    -------
    transformed_image : PIL Image object
        The transformed image
    transformed_label : PIL Image object
        The transformed label

    """
    #set cropping parameters
    random_crop_size = image_dim
    random_crop_scale = (0.5,1.5)

    #If the identity flag is true, only crop the image and label
    if identity_map:
        return image.resize((random_crop_size,random_crop_size)), label.resize((random_crop_size,random_crop_size))

    else:
        #set image transformation
        image_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomResizedCrop(size = random_crop_size, scale = random_crop_scale),

            ### Image only transforms ###
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2),
            transforms.GaussianBlur(21)

        ])

        #set the label transformation (exclude the colour scaling)
        label_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomResizedCrop(size = random_crop_size, scale = random_crop_scale),
        ])

        #Create a transformer object
        transformer = CustomTransform(image_transform, label_transform)

        #transform the images and labels
        augmented_image, augmented_label = transformer(image, label)

        return augmented_image, augmented_label

def open_datapoint(image_path,label_path):
    """

    Parameters
    ----------
    image_path : string
        The image filename, including the path
    label_path : string
        The image filename, including the path

    Returns
    -------
    image : PIL Image object
        The image found in the specified path
    label : PIL Image object
        The label found in the specified path

    """

    #Open the images in the specified paths
    image = Image.open(image_path)
    label = Image.open(label_path)

    return image, label

def save_new_datapoint(image,label,image_path,label_path,filename):
    """

    Parameters
    ----------
    image : PIL Image object
        The image to be saves
    label : PIL Image object
        The label to be saves
    image_path : string
        The directory you whish to save the image in
    label_path : string
        The directory you whish to save the label in
    filename : string
        The name of the files (excluding the extension)

    Returns
    -------
    None.

    """

    #Make the directories if they don't exist
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    #Try to save the files
    try:
        #Save the files
        image.save(image_path+filename+'.jpg')
        label.save(label_path+filename+'.png')

    #catch OS error
    except OSError as e:
        #If a specific file fails to save, warn the user
        print(f"{filename} failed to save \n {e}")

def convert_to_h5(image_dir, label_dir, database_file, image_shape = 300):
    """
    Parameters
    ----------
    image_dir : str
        The directory of your image files.
    label_dir : str
        The directory of your label files
    database_file : str
        The name of the .h5 database file you whish to save the dataset in.
    image_shape : int, optional
        The image dataset image dimensions. The default is 300.

    Returns
    -------
    None.

    """
    #find the filanames of the images
    filenames = [f.replace('.jpg','') for f in list(os.listdir(image_dir))]

    # Create an HDF5 file
    with h5py.File(database_file, "w") as h5f:

        #specify the image and label sizes
        img_shape = (len(filenames), image_shape, image_shape, 3)
        label_shape = (len(filenames), image_shape, image_shape)

        #create the h5 datasets
        image_dataset = h5f.create_dataset("images", img_shape, dtype=np.uint8)
        filename_dataset = filename_dataset = h5f.create_dataset("filenames", (len(filenames),), dtype=h5py.string_dtype())
        label_dataset = h5f.create_dataset("labels", label_shape, dtype=np.uint8)

        # Store images, labels and filenames in the database
        for i, file in tqdm(enumerate(filenames), total=len(filenames),desc='Converting to .h5 file', unit = ' Images'):

            #Open the images and labels
            img = Image.open(os.path.join(image_dir, file +'.jpg'))
            label = Image.open(os.path.join(label_dir, file +'.png'))

            try:
                #Store the images and labels in the .h5 file
                image_dataset[i] = np.array(img)
                label_dataset[i] = np.array(label)
                filename_dataset[i] = file
            except:
                print(f"Conversion of {file} failed")


if __name__ == '__main__':

    # Define augmented data directory
    augmented_data_dir = '../augmented-dataset/TrainVal/color/'
    augmented_label_dir = '../augmented-dataset/TrainVal/label/'

    #Set the number of datapoints
    augmentations_per_datapoints = 4

    #Fetch filenames in dataset directory
    data_files, label_files, filenames = fetch_dataset_filenames()

    #Itterate through filenames
    for i in tqdm( range(0,len(data_files)), desc = 'Transforming dataset', unit = ' Images'):

        #Open image and label file
        image, label = open_datapoint(data_files[i],label_files[i])

        #crop image using the identity map flag for the transform
        cropped_image, cropped_label = transform_datapoint(image, label, identity_map=True)

        #save the cropped datapoint
        save_new_datapoint(cropped_image,cropped_label,augmented_data_dir,augmented_label_dir,f"{filenames[i]}(0)")

        #itterate through augmented datapoints
        for j in range(0,4):

            #augment image
            augmented_image, augmented_label = transform_datapoint(image, label)

            #save image
            save_new_datapoint(augmented_image,augmented_label,augmented_data_dir,augmented_label_dir,f"{filenames[i]}({j+1})")

    #convert all images in augmented dataset directory to .h5 file
    convert_to_h5('../augmented-dataset/TrainVal/color/', '../augmented-dataset/TrainVal/label/', '../augmented-dataset/TrainVal.h5')

    # Define augmented test data directory
    augmented_test_data_dir = '../augmented-dataset/Test/color/'
    augmented_test_label_dir = '../augmented-dataset/Test/label/'

    #Fetch filenames in test dataset directory
    test_data_files, test_label_files, test_filenames = fetch_dataset_filenames(image_directory = '../Dataset/Test/color/', label_directory = '../Dataset/Test/label/')

    #Itterate through test filenames
    for i in tqdm( range(0,len(test_data_files)), desc = 'Cropping test dataset', unit = ' Images'):

        #Open image and label file
        image, label = open_datapoint(test_data_files[i],test_label_files[i])

        #crop image using the identity map flag for the transform
        cropped_image, cropped_label = transform_datapoint(image, label, identity_map=True)

        #save the cropped datapoint
        save_new_datapoint(cropped_image,cropped_label,augmented_test_data_dir,augmented_test_label_dir,f"{test_filenames[i]}")

    #convert all test images in augmented dataset directory to .h5 file
    convert_to_h5('../augmented-dataset/Test/color/', '../augmented-dataset/Test/label/', '../augmented-dataset/Test.h5')
