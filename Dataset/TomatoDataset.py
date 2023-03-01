import os

import torch
from torchvision.transforms import transforms

from Config.DatasetConfig import width,hight
from torch.utils.data import Dataset
import pandas as pd
import cv2
class TomatoDataet(Dataset):
    def __init__(self, opt,data_precent):
        super(TomatoDataet,self).__init__()
        self.opt = opt

        path_to_scv = os.path.join(opt.root_path,opt.csv_path)
        self.df = pd.read_csv(path_to_scv)
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

        img_path = self.df.iloc[index]["path"]
        label = self.df.iloc[index]["label"]


        path_to_img = os.path.join(self.opt.root_path,self.opt.root_dataset, img_path)
        img = cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img,(width,hight))

        img_norm = cv2.normalize(img_resized,None,0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)
        # mask_norm_t = torch.from_numpy(mask_norm)
        img_norm_t = (self.input_transform(img_norm) + 1)/2

        x = torch.tensor([[label]])


        return img_norm_t, x

    def test(self,index):
        input, output = self[index]
