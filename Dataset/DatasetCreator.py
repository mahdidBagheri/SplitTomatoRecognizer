import glob
import os

import pandas as pd
import cv2

def create_csv():
    df = pd.DataFrame(columns=["path", "label"])
    dataset_root = "Dataset"
    root_Address = os.getcwd()
    rel_Address_split = "singles_clean/singles_clean/big/split"
    rel_Address_no_split = "singles_clean/singles_clean/big/no_split"

    no_split_address = os.path.join(root_Address,rel_Address_no_split)
    no_plit_paths = glob.glob(no_split_address + "/*.jpg")

    split_address = os.path.join(root_Address,rel_Address_split)
    split_paths = glob.glob(split_address + "/*.jpg")

    for i, p in enumerate(no_plit_paths):
        img = cv2.imread(p)

        #augment
        #///
        #augment
        new_path = os.path.join(dataset_root,f"no_split_{i}.jpg")
        new_df = pd.DataFrame({"path":[new_path], "label":[0.0]})
        df = pd.concat((df,new_df), ignore_index=True)
        cv2.imwrite(new_path,img)

    for i, p in enumerate(split_paths):
        img = cv2.imread(p)

        # augment
        # ///
        # augment

        new_path = os.path.join(dataset_root,f"split_{i}.jpg")
        new_df = pd.DataFrame({"path": [new_path], "label": [1.0]})
        df = pd.concat((df, new_df), ignore_index=True)
        cv2.imwrite(new_path, img)

    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("data.csv", index_label=False)



if(__name__=="__main__"):
    create_csv()