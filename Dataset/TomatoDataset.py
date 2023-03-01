import os

import torch
from torchvision.transforms import transforms

from Config.DatasetConfig import width,hight
from torch.utils.data import Dataset
import pandas as pd
import cv2
class TomatoDataet(Dataset):
    def __init__(self, opt, csv_path,data_precent):
        super(TomatoDataet,self).__init__()
        self.opt = opt

        path_to_scv = os.path.join(opt.root_path,opt.root_dataset, csv_path)
        self.df = pd.read_csv(path_to_scv)
        self.df = self.df.loc[self.df['scale'] > 0.6]
        self.df = self.df.iloc[int(len(self.df)*data_precent[0]):int(len(self.df)*data_precent[1])]

        transform_list_input = [transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]
        self.input_transform = transforms.Compose(transform_list_input)

        transform_list_output = [transforms.ToTensor()]
        self.output_transform = transforms.Compose(transform_list_output)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        assert index <= len(self)

        combined_path = self.df.iloc[index]["img_name"]
        mask_path = self.df.iloc[index]["mask_name"]
        position = self.df.iloc[index]["position"]
        label = self.df.iloc[index]["label"]
        # print(label)
        scale = self.df.iloc[index]["scale"]

        path_to_mask = os.path.join(self.opt.root_path,self.opt.root_dataset, mask_path)
        mask_img = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img,(width,hight))

        mask_norm = cv2.normalize(mask_img,None,0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)
        # mask_norm_t = torch.from_numpy(mask_norm)
        mask_norm_t = (self.input_transform(mask_norm) + 1)/2

        path_to_combined = os.path.join(self.opt.root_path,self.opt.root_dataset, combined_path)
        combined_img = cv2.imread(path_to_combined)
        combined_img = cv2.resize(combined_img,(width,hight))
        combined_norm = cv2.normalize(combined_img,None,0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)
        # combined_norm_t = torch.from_numpy(combined_norm)
        combined_norm_t = self.output_transform(combined_norm)

        return combined_norm_t, mask_norm_t

    def test(self,index):
        input, output = self[index]
