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
    parser.add_argument("--cuda", default=False, type=bool)

    opt = parser.parse_args()

    train_dataset = TomatoDataset(opt, train_fraction)
    test_dataset = TomatoDataset(opt, test_fraction)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = TomatoNN()
    if(torch.cuda.is_available() and opt.cuda):
        model = model.cuda()
        pass

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    learner = Learner(opt, model, loss, optimizer, train_loader, test_loader)
    for epoch in range(epochs):
        train_results = learner.run_epoch(epoch,val=False)
        test_results = learner.run_epoch(epoch,val=True)
        scheduler.step()

        if(epoch % save_period == 0):
            learner.save(path=f"{opt.output}/model_{epoch}.pt")
