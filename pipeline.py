import numpy as np 
import pandas as pd 
import re

from imageio import imread

import cv2 as cv 

from image_processing import get_mask_from_encoding

file_height = 1400
file_width = 2100
input_height = 256
input_width = 384
data_folder = 'understanding_clouds'
def separate_train_val(labels_df,val_ratio=0.1):
    labs = pd.DataFrame(pd.unique(labels_df['Image_Label'].apply(lambda x: re.match("(\w+.jpg)_[a-zA-Z]+",x)[1])))
    labs.columns = ['Image_Label']
    val_ = labs.sample(frac=val_ratio)
    train_ = labs.loc[~labs.index.isin(val_.index)]
    train = pd.DataFrame(columns=['Image_Label'])
    val = pd.DataFrame(columns=['Image_Label'])

    for lab_type in label_names:
        train_t = train_['Image_Label'] + '_' + lab_type
        train_t.columns = ['Image_Label']
        train_t =  pd.DataFrame(train_t)
        train = pd.concat([train,train_t])

        val_t = val_['Image_Label'] + '_' + lab_type
        val_t.columns = ['Image_Label']
        val_t =  pd.DataFrame(val_t)
        val = pd.concat([val,val_t])
    train = labels_df.loc[labels_df['Image_Label'].isin(train['Image_Label'])]
    val = labels_df.loc[labels_df['Image_Label'].isin(val['Image_Label'])]
    return train, val    

label_names = ['Fish','Flower','Gravel','Sugar']

def custom_image_generator(bs, labels, path_to_data, predicted_class,mode="train", 
                           aug=None, input_height=input_height, input_width=input_width,
                           prefix=''):
    
    assert predicted_class in label_names
    folder_path = path_to_data
    # get the appropriate dataframe 
    if mode == "train" or mode == "validation":
        dataframe_name = "train.csv"
        df = labels.copy()
        
        df = df.loc[df['Image_Label'].str.endswith(predicted_class)]
        # We only use images where there is the label of interest to have a balanced dataset
        df = df.loc[df['EncodedPixels'].notnull()]
    elif mode == "test":
        dataframe_name = "sample_submission.csv"
    
        df = pd.read_csv(dataframe_name)
        df = df.loc[df['Image_Label'].str.endswith(predicted_class)]
    # loop indefinitely
    i = 0 
    while True:
        # initialize our batches of images and labels
        images = []
        labels = []
        # keep looping until we reach our batch size
        while len(images) < bs:
            try:
                line = df.iloc[i]
            except:
                print(i)
            # attempt to read the next line of the CSV file
             
            # check to see if the number of the current line is equal to 
            # the length of the dataframe, indicating we have reached the end
            # of the dataframe
            if i == len(df):
                # reset the file pointer to the beginning of the file
                # and re-read the line
                i = 0 
            
            if mode == "test":
                break
                
            # extract the label and construct the image
            label = line['EncodedPixels']
            raw_mask = get_mask_from_encoding(label,
                                             image_height=file_height,
                                             image_width=file_width)
            resized_mask = np.expand_dims(cv.resize(raw_mask,(input_width,input_height)),-1)
            image_path = prefix+folder_path+line['Image_Label']
            raw_im = imread(re.match("(.*\/\w+.jpg)_[a-zA-Z]+",image_path)[1])/256     
            resized_im = cv.resize(raw_im,(input_width,input_height))
            
            # update our corresponding batches lists
            images.append(resized_im)
            labels.append(resized_mask) 
            i += 1

        # if the data augmentation object is not None, apply it
        if aug is not None:
            (images, labels) = next(aug.flow(np.array(images),labels, batch_size=bs))

        # yield the batch to the calling function
        yield (np.array(images), np.array(labels))