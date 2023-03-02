import copy
import glob
import math
import os

import pandas as pd
import cv2

# rot,
def resize_img(img, max_size):
    if(img.shape[0] * img.shape[1] > max_size):
        scale = math.sqrt(max_size) / min(img.shape[0], img.shape[1])
        img = cv2.resize(img, (int(img.shape[0]*scale), int(img.shape[0]*scale)))
    return img


def create_csv():
    df = pd.DataFrame(columns=["path", "label"])
    dataset_root = "Dataset"
    root_Address = os.getcwd()
    rel_Address_split = "singles_clean/singles_clean/*/split"
    rel_Address_no_split = "singles_clean/singles_clean/*/no_split"

    no_split_address = os.path.join(root_Address,rel_Address_no_split)
    no_plit_paths = glob.glob(no_split_address + "/*.jpg")

    split_address = os.path.join(root_Address,rel_Address_split)
    split_paths = glob.glob(split_address + "/*.jpg")

    for i, p in enumerate(no_plit_paths):
        img = cv2.imread(p)
        img = resize_img(img, max_size=512*512)
        #augment
        image = copy.deepcopy(img)
        for j in range(4):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE, image)

            new_path = os.path.join(dataset_root,f"no_split_{i}_{j}.jpg")
            new_df = pd.DataFrame({"path":[new_path], "label":[0.0]})
            df = pd.concat((df,new_df), ignore_index=True)
            cv2.imwrite(new_path,image)

    for i, p in enumerate(split_paths):
        img = cv2.imread(p)
        img = resize_img(img, max_size=512*512)

        # augment
        image = copy.deepcopy(img)
        for j in range(3):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            new_path = os.path.join(dataset_root,f"split_{i}_{j}.jpg")
            new_df = pd.DataFrame({"path": [new_path], "label": [1.0]})
            df = pd.concat((df, new_df), ignore_index=True)
            cv2.imwrite(new_path, image)

    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("data.csv", index_label=False)



if(__name__=="__main__"):
    create_csv()