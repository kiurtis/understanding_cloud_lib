import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def get_images_with_label(df_labels,label,n_labels=3):
    df_labels_f = df_labels.loc[df_labels['Image_Label'].str.endswith(label)]
    row_nums = np.random.choice(len(df_labels_f), n_labels, replace=False)
    n_row = n_labels//3
    n_col = 3
    fig, ax = plt.subplots(n_row,n_col,figsize=(30,20))
    for i in range(n_row):
        for j in range(n_col):
            im_path = train_path+df_labels_f.iloc[row_nums[i*n_col+j],0]
            ax[i][j].imshow(imread(re.match("(.*\/\w+.jpg)_[a-zA-Z]+",im_path)[1]))
    fig.suptitle('Random picture containing the "'+ label + '" label', fontsize=16)
    fig.tight_layout()
    return im_path