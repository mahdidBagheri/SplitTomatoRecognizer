import copy
from screeninfo import get_monitors
import numpy as np
import torch
import torchvision
import argparse
import cv2
from models.TomatoNN import TomatoNN
from torchvision.transforms import transforms
from Config.DatasetConfig import width,hight

def disp(result_img):
    monitors = get_monitors()
    monitor = monitors[0]
    H = monitor.height
    W = monitor.width

    scale = min(H/result_img.shape[0], W/result_img.shape[1]) * 0.8
    result_img = cv2.resize(result_img, (int(result_img.shape[1]*scale), int(result_img.shape[0]*scale)))
    cv2.imshow("result", result_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--img_path")
    parser.add_argument("--save", default="my_beloved_tomato.png")
    parser.add_argument("--cuda", default=False, action='store_true')
    opt = parser.parse_args()

    model = TomatoNN()
    model.load_state_dict(torch.load(opt.model)["model"])
    model.eval()

    src_img = cv2.imread(opt.img_path)

    img = copy.deepcopy(src_img)
    img = cv2.resize(img, (width, hight))
    img_norm = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    # combined_norm_t = torch.from_numpy(combined_norm)
    transform_list_output = [transforms.ToTensor()]
    output_transform = transforms.Compose(transform_list_output)

    img_t = output_transform(img_norm)[None,:,:,:]

    if(torch.cuda.is_available() and opt.cuda):
        combined_norm_t = img_t.cuda()
        model = model.cuda()

    with torch.no_grad():
        output = model(img_t)

    output = output.cpu().detach().numpy().reshape(1).item()

    # visualization
    result_bar = np.zeros((int(src_img.shape[0]/5),src_img.shape[1],3), dtype=np.uint8)
    if(output < 0.3):
        text = "non-split"
    else:
        text = "split"

    result_bar = cv2.putText(result_bar, text, (int(src_img.shape[1]/5),int(result_bar.shape[0]/2)), fontFace=1, fontScale=int(src_img.shape[1]/100), thickness=int(src_img.shape[1]/100), color=(255,0,0))
    result_img = np.concatenate((src_img, result_bar), axis=0)
    cv2.imwrite(opt.save, result_img)
    disp(result_img)


    a = 0


