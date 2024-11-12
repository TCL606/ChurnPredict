# 银行客户流失预测

---

## 任务说明

见 [instructions.pdf](instructions.pdf)。

## 算法设计

我希望设计一个多层感知机模型来实现逻辑回归，模型的输入是用户的特征向量，输出是用户流失与否的概率。模型的定义和前向传播过程如下：

~~~python
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
~~~

在模型初始化函数中，`input_size` 是用户特征向量的维度，即为模型的输入维度；`hidden_size` 为模型隐藏层维度。线性层之间使用激活函数 `ReLU` 来赋予模型非线性能力；模型的最后将隐层向量映射到数字，用 `sigmoid` 函数将该数字转换为用户流失与否的概率。

在模型的前向传播过程中，我们使用多层感知机模型和 `sigmoid` 函数将用户输入特征映射到概率，通过一定的损失函数计算损失，用于反向传播。损失函数可以是 `BCE Loss` 或 `MSE Loss`。

假设模型输出值为 `x`，经过  `sigmoid` 函数后得到 `p`，设模型标签为 `l`，则 `BCE Loss` 的理论表达式为：

$\text{BCE Loss}=−(l⋅log⁡(p)+(1−l)⋅log⁡(1−p))$

`MSE Loss` 的理论表达式为：

$\text{MSE Loss}=(l−p)^2$

## 数据预处理

数据预处理阶段，我主要解决数据清洗、数值特征的归一化和非数值数据的处理，目标是得到用户相关的特征向量。

- 数据清洗：我主要将数据去除无用的维度，不将其用于训练。观察数据可知，`train_idx` 这一列指的是用户的 `id`，理论上对预测用户是否流失没有任何意义，应该将其去除。同时 `Attrition_Flag` 这一列是最终的标签，同样不应该用于训练。
- 数值特征的归一化：对于数值特征，我计算该属性所有用户的均值与标准差，通过先减均值再除以标准差的方法对数据归一化
- 非数值特征的处理：非数值特征方面，我希望将其映射为多维的 `one-hot` 向量。首先，对于某一属性，我统计出现过不同的字符串个数，将其一一映射到一个数字，用于生成 `one-hot` 向量。举例来说，例如对于 `Gender` 属性，总共有 `F` 和 `M` 两类，分别将其映射为 0 和 1，最后生成的 `one-hot` 向量分别为 `[1, 0]` 和 `[0, 1]`。

数据预处理的代码如下：

~~~python
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
~~~

通过预处理好的数据，我利用其来构建 `Dataset`，代码如下：

~~~python
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
~~~

上述代码中，我将每个用户的所有一维归一化数值特征和所有多维 `one-hot` 非数值特征全部连接到一起，组成该用户的总的特征向量。

## 训练过程

我将所有数据随机划分为 7000 条训练集和 1101 条测试集，代码如下:

~~~python
data = preprocess(args.csv)
random.shuffle(data)
train_data, val_data = data[:7000], data[7000:] # split train and val data
~~~

使用 `Adam` 优化器进行梯度下降。每一步更新的代码如下，其中 `optimizer` 为 `Adam` 优化器：

~~~python
model.train()
optimizer.zero_grad()
_, loss = model(**batch)
loss.backward()
optimizer.step()
~~~

其中，`loss` 为针对标签的 `BCE Loss` ，标签为 1 或 0，分别对应客户是否流失。

训练过程中，每 100 步进行一次模型评测，记录 `loss` 和相关指标的变化。模型评测的代码如下：

~~~python
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
~~~

在函数 `eval_model` 中，统计 `precision`、`recall` 、`accuracy` 等指标。

整个训练过程中，我们尝试训练 40 `epochs`，其中学习率为 `1e-4`。由于统计单步损失函数变化较大，因此损失函数曲线的绘制，我使用每 100 步的平均损失来作图。整个过程中训练集的损失函数变化如下：

<img src="output/bce_40epoch/avg_loss.png" alt="avg_loss" style="zoom:72%;" />

训练过程中验证集上的 `precision` 曲线为

<img src="output/bce_40epoch/precision.png" alt="precision" style="zoom:72%;" />

训练过程中验证集上的 `recall` 曲线为

<img src="output/bce_40epoch/recall.png" alt="recall" style="zoom:72%;" />

训练过程中验证集上的 `accuracy` 曲线为

<img src="output/bce_40epoch/acc.png" alt="acc" style="zoom:72%;" />

可以看到，在训练 0-15000 步时，模型欠拟合，此时模型在训练中在验证集上性能显著提高。训练至 20000 步左右时，`accuracy` 就不再上升了，模型已收敛。最后在模型训练了 40 `epochs` 后，`accuracy` 仍然保持在 `93%` 左右，说明模型收敛。暂未观察到过拟合现象。

训练 40 `epochs` 后，模型的结果为：

~~~
Final Precision 0.9622844827586207
Final Recall 0.9550802139037433
Final F1 0.9586688137412775
Final Acc 0.9300635785649409
~~~



## 回归模型建模

建模为回归时，损失函数可换为 `MSE Loss`，与标签直接计算均方误差。

训练集的损失函数变化如下：

<img src="output/mse_40epoch/avg_loss.png" alt="avg_loss" style="zoom:72%;" />

训练过程中验证集上的 `precision` 曲线为

<img src="output/mse_40epoch/precision.png" alt="precision" style="zoom:72%;" />

训练过程中验证集上的 `recall` 曲线为

<img src="output/mse_40epoch/recall.png" alt="precision" style="zoom:72%;" />

训练过程中验证集上的 `accuracy` 曲线为

<img src="output/mse_40epoch/acc.png" alt="precision" style="zoom:72%;" />

训练 40 `epochs` 后，模型的结果为：

~~~
Final Precision 0.961456102783726
Final Recall 0.960427807486631
Final F1 0.9609416800428037
Final Acc 0.9336966394187103
~~~



分类模型和回归模型模型整体的变化趋势基本一致，最后结果也非常接近。

可能的原因是：

- 该数据集的用户特征可能在高维空间下，本质上是一个容易区分的二分类问题，那么分类模型和回归模型的最终结果会相似。所以即使是回归模型，也很容易学到一个清晰的决策边界，使得它在分类任务上表现良好。无论是哪个模型，最终分类的准确率在 `93%` 左右，是一个比较高的值，也从侧面佐证了该二分类问题是相对容易区分的。
- 同时，无论是分类模型还是回归模型，模型结构都是相同的，只是优化的损失函数不同。由于模型的输出都是一个概率数字，且两种损失函数在二分类问题下差异并不大，因此模型的优化趋势会相似。