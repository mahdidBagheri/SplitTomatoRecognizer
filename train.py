import glob
import os
from tqdm import tqdm
import torch
import torchvision
from Dataset.TomatoDataset import TomatoDataset
import argparse
from Config.ModelConfig import epochs
from Config.LearnerConfig import save_period, batch_size, lr, gamma, train_fraction, test_fraction
from models.TomatoNN import TomatoNN
from models.TomatoLearner import Learner
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR


if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--root_path", default=os.getcwd())
    parser.add_argument("--root_dataset", required=True)
    parser.add_argument("--batch_size", default=batch_size)
    parser.add_argument("--lr", default=lr)
    parser.add_argument("--epochs", default=epochs)
    parser.add_argument("--output")
    parser.add_argument("--cuda", default=False, action='store_true')
    parser.add_argument("--resume", default=False, action='store_true')

    opt = parser.parse_args()

    train_dataset = TomatoDataset(opt, train_fraction)
    test_dataset = TomatoDataset(opt, test_fraction)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = TomatoNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss = torch.nn.MSELoss()
    init_epoch = 0
    if (opt.resume):
        chk_pt_path = "output/*.pt"
        models_path = glob.glob(chk_pt_path)
        checkpoint = torch.load(models_path[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    if(torch.cuda.is_available() and opt.cuda):
        model = model.cuda()


    scheduler = ExponentialLR(optimizer, gamma=gamma)


    learner = Learner(opt, model, loss, optimizer, train_loader, test_loader, init_epoch)
    for epoch in range(init_epoch,epochs):
        train_results = learner.run_epoch(epoch,val=False)
        test_results = learner.run_epoch(epoch,val=True)
        scheduler.step()

        learner.save(path=f"{opt.output}/last.pt")
        if(epoch % save_period == 0):
            learner.save(path=f"{opt.output}/model_{epoch}.pt")
