import numpy as np 
import cv2 as cv


kernel = np.ones((20,20),np.uint8)


def safe_nan(x):
    try: 
        res = np.isnan(x)
    except:
        res = False
    return res

def morph_transform(mask : np.array):
    floated = np.float32(mask)
    transformed = cv.morphologyEx(floated, cv.MORPH_OPEN, kernel)
    transformed = cv.dilate(transformed,kernel,iterations = 1)
    return transformed

def get_encoding_from_mask(mask,height):
    grad = mask[1:,:]-mask[:-1,:]

    # We add the columns starting with a mask
    first_row = np.zeros(mask.shape[1])
    first_row[(mask[0] ==1)] = 1
    grad = np.vstack([first_row,grad])
    
    # We also must consider that when  the mask continues to the end, then it's an ending point (in the grad computation it does not appear)
    grad[-1,(mask[-1] ==1)] = -1
    
    starting_points = np.where(grad==1)
    ending_points = np.where(grad==-1)

    starting_points =[x for x in zip(starting_points[0],starting_points[1])]
    starting_points = sorted(starting_points,key=lambda x:x[1])
    starting_points_raveled = [x[1]*height+x[0] for x in starting_points]

    ending_points =[x for x in zip(ending_points[0],ending_points[1])]
    ending_points = sorted(ending_points,key=lambda x:x[1])
    ending_points_raveled = [x[1]*height+x[0] for x in ending_points]
    lengths = [x-y for x,y in zip(sorted(ending_points_raveled),sorted(starting_points_raveled))]
    encoded = ' '.join([str(x[0]) +' '+ str(x[1]) for x in zip(starting_points_raveled,lengths)])
    return encoded

def get_mask_from_encoding(encoded, image_height, image_width):
    mask_res = np.zeros((image_height,image_width))    
    # Case when there is no labels
    if safe_nan(encoded):
        return mask_res
    
    # Case when there is labels
    encoded_labels_list = encoded.split(' ')
    couples = [(int(k),int(l)) for k,l in zip(encoded_labels_list[::2], encoded_labels_list[1::2])]
    for cp in couples: 
        i = (cp[0])%(image_height)
        j = (cp[0])//(image_height)
        mask_res[i:i+cp[1],j] =1
        #print(i,j)
    return mask_res