import os
from tqdm import tqdm
import torch
import torchvision
from Dataset import TomatoDataset
import argparse
from Config.ModelConfig import epochs
from Config.LearnerConfig import save_period, batch_size, lr, gamma
from models.TomatoNN import TomatoNN
from models.TomatoLearner import TomatoLearner
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--root_path", default=os.getcwd())
    parser.add_argument("--root_dataset", required=True)
    parser.add_argument("--batch_size", default=batch_size)
    parser.add_argument("--lr", default=lr)
    parser.add_argument("--epochs", default=epochs)
    parser.add_argument("--output")

    opt = parser.parse_args()

    train_dataset = TomatoDataset.TomatoDataset(opt)
    test_dataset = TomatoDataset.TomatoDataset(opt)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = TomatoNN()
    if(torch.cuda.is_available()):
        model = model.cuda()

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    learner = TomatoLearner(model, loss, optimizer, train_loader, test_loader)
    for epoch in range(epochs):
        train_results = learner.run_epoch(epoch,val=False)
        test_results = learner.run_epoch(epoch,val=True)
        scheduler.step()

        if(epoch % save_period == 0):
            learner.save(path=f"{opt.output}/model_{epoch}.pt")
