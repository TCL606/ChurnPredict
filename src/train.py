import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import argparse
import random
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import List
import json

class LogRegModel(nn.Module):
    # input_size: user feature dim; hidden_size: hidden dim
    def __init__(self, input_size=38, hidden_size=64, criterion="bce"): 
        super(LogRegModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        if criterion == "bce":
            self.criterion = nn.BCELoss()
        elif criterion == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise Exception(f"Invalid criterion: {criterion}")

    
    def forward(self, input, label=None):
        x = self.mlp(input)
        x = x.squeeze()
        x = torch.sigmoid(x)
        if label is None:
            return x, None
        else:
            loss = self.criterion(x, label)
            return x, loss


class ChurnDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.tensor_data = []
        for item in data:
            item_dim = []
            for key in item.keys():
                if key == "train_idx": # do not use user id for training
                    continue
                if key == "Attrition_Flag": # do not use the label for training
                    gt_label = item[key].item()
                    break
                if isinstance(item[key], List):
                    one_hot_label = torch.tensor(item[key][0])
                    one_hot_tensor = F.one_hot(one_hot_label, item[key][1])
                    item_dim.append(one_hot_tensor.to(torch.float32))
                else:
                    item_dim.append(torch.tensor([item[key]]).to(torch.float32))
            item_tensor = torch.cat(item_dim) # concat all feature of one user
            self.tensor_data.append({
                "input": item_tensor,
                "label": gt_label
            })

    def __len__(self):
        return len(self.tensor_data)
    
    def __getitem__(self, idx):
        return self.tensor_data[idx]
    
    @classmethod
    def collate(cls, items):
        input = [it["input"] for it in items]
        input = torch.stack(input)

        label = [it['label'] for it in items]
        label = torch.tensor(label, dtype=torch.float32)

        return {"input": input, "label": label}
    
def eval_model(model, val_loader):
    model.eval()
    true_crt, false_crt, true_wrong, false_wrong, total = 0, 0, 0, 0, 0
    for batch in val_loader:
        label = batch.pop("label")
        x, _ = model(**batch)
        x = x > 0.5
        label = label == 1

        # to compute precision and recall
        true_crt += torch.sum((x == label) & (x == True)).item()
        false_crt += torch.sum((x == label) & (x == False)).item()
        true_wrong += torch.sum((x != label) & (x == True)).item()
        false_wrong += torch.sum((x != label) & (x == False)).item()
        total += x.size(0)

    precision = true_crt / (true_crt + true_wrong) if (true_crt + true_wrong) > 0 else 0
    recall = true_crt / (true_crt + false_wrong) if (true_crt + false_wrong) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    acc = (true_crt + false_crt) / total
    return precision, recall, f1, acc

def train_model(model, train_loader, val_loader, lr, num_epochs=5, log_steps=100):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses, avg_losses, tmp_losses, precisions, recalls, f1s, accs = [], [], [], [], [], [], []
    total_steps = num_epochs * len(train_loader)
    step = 0

    with tqdm(total=total_steps, desc='Training') as pbar:
        for _ in range(num_epochs):
            for batch in train_loader:
                model.train()
                optimizer.zero_grad()
                _, loss = model(**batch)
                loss.backward()
                optimizer.step()

                if step % log_steps == 0:
                    precision, recall, f1, acc = eval_model(model, val_loader)
                    if len(tmp_losses) != 0:
                        avg_losses.append([step, sum(tmp_losses) / len(tmp_losses)]) # record average loss
                        tmp_losses = []
                    precisions.append([step, precision])
                    recalls.append([step, recall])
                    f1s.append([step, f1])
                    accs.append([step, acc])


                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                step += 1
                losses.append([step, loss.item()])
                tmp_losses.append(loss.item())

        precision, recall, f1, acc = eval_model(model, val_loader)
        if len(tmp_losses) != 0:
            avg_losses.append([step, sum(tmp_losses) / len(tmp_losses)])
            tmp_losses = []
        precisions.append([step, precision])
        recalls.append([step, recall])
        f1s.append([step, f1])
        accs.append([step, acc])

    return losses, avg_losses, precisions, recalls, f1s, accs

def preprocess(csv_file):
    data = pd.read_csv(csv_file)
    res = []
    for i in range(len(data['train_idx'])): # extract user's all elements
        item = {}
        for key in data.keys():
            item[key] = data[key][i]
        res.append(item)

    for key in data.keys():
        if key == "train_idx" or key == "Attrition_Flag": # Do not use user id and the label
            continue
        if isinstance(res[0][key], str): # if str attribute
            exist_dict = {}
            cnt = 0
            for item in res:
                if item[key] not in exist_dict:
                    exist_dict[item[key]] = cnt
                    cnt += 1
            for i in range(len(res)):
                res[i][key] = [exist_dict[res[i][key]], cnt]

        else: # else numeric attribute
            num_lst = [it[key] for it in res]
            mean = np.mean(num_lst)
            std = np.std(num_lst, ddof=1)
            for i in range(len(res)):
                res[i][key] = (res[i][key] - mean) / std

    return res

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 2')
    parser.add_argument('--csv', type=str, default="/Users/bytedance/Desktop/课程/机器学习/第一次编程作业/supply_chain_train.csv")
    parser.add_argument('--output_root', type=str, default="/Users/bytedance/Desktop/课程/机器学习/第一次编程作业/output")
    parser.add_argument('--output_name', type=str, default="debug")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--criterion', type=str, default="bce")
    args = parser.parse_args()

    set_seed(args.seed)

    data = preprocess(args.csv)
    random.shuffle(data)
    train_data, val_data = data[:7000], data[7000:] # split train and val data
    train_id_lst = [it["train_idx"].item() for it in train_data]
    val_id_lst = [it["train_idx"].item() for it in val_data]
    print(len(train_id_lst), len(val_id_lst))
    
    train_dataset, val_dataset = ChurnDataset(train_data), ChurnDataset(val_data)
    user_dim = train_dataset[0]['input'].shape[0]

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=ChurnDataset.collate)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=ChurnDataset.collate)

    model = LogRegModel(user_dim, args.hidden_dim, args.criterion) # model definition

    if not args.do_test:
        output_dir = os.path.join(args.output_root, args.output_name)
        os.makedirs(output_dir, exist_ok=True)
        data_split = {"train_id_lst": train_id_lst, "val_id_lst": val_id_lst}
        with open(os.path.join(output_dir, "data_split.json"), 'w') as fp:
            json.dump(data_split, fp, indent=4)

        losses, avg_losses, precisions, recalls, f1s, accs = train_model(model, train_loader=train_loader, val_loader=val_loader, lr=args.lr, num_epochs=args.epochs)
        print("Final Precision", precisions[-1][1])
        print("Final Recall", recalls[-1][1])
        print("Final F1", f1s[-1][1])
        print("Final Acc", accs[-1][1])

        torch.save(model.state_dict(), os.path.join(output_dir, "model.bin"))

        # plot figures
        plt.figure()
        plt.plot([it[0] for it in losses], [it[1] for it in losses])
        plt.savefig(os.path.join(output_dir, "loss.png"))

        plt.figure()
        plt.plot([it[0] for it in avg_losses], [it[1] for it in avg_losses])
        plt.savefig(os.path.join(output_dir, "avg_loss.png"))

        plt.figure()
        plt.plot([it[0] for it in precisions], [it[1] for it in precisions])
        plt.savefig(os.path.join(output_dir, "precision.png"))

        plt.figure()
        plt.plot([it[0] for it in recalls], [it[1] for it in recalls])
        plt.savefig(os.path.join(output_dir, "recall.png"))

        plt.figure()
        plt.plot([it[0] for it in f1s], [it[1] for it in f1s])
        plt.savefig(os.path.join(output_dir, "f1.png"))

        plt.figure()
        plt.plot([it[0] for it in accs], [it[1] for it in accs])
        plt.savefig(os.path.join(output_dir, "acc.png"))

    else:
        ckpt = torch.load(args.ckpt, weights_only=True)
        model.load_state_dict(ckpt)
        precision, recall, f1, acc = eval_model(model, val_loader)
        print("Eval Precision:", precision)
        print("Eval Recall:", recall)
        print("Eval F1:", f1)
        print("Eval Acc:", acc)
    