import numpy as np
import torch
from numpy import float32, float64
from torch.cuda import device
from config.node_info import *
from utils.eth_instance import EthInstance
import json
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
from utils import arg_parser
import os


class NeuralNet(nn.Module):
    def __init__(self, input_num, hidden1_num, hidden2_num, output_num):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_num, hidden1_num), nn.ReLU(True))
        self.fc2 = nn.Sequential(
            nn.Linear(hidden1_num, hidden2_num), nn.ReLU(True))
        self.fc3 = nn.Sequential(nn.Linear(hidden2_num, output_num))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# load model weights

def loadWightsIntoLocalModel(model, weights_path, dwload_weights):
    with open(weights_path, 'r') as f:
        weights_structure = json.load(f)

    fc1_w = weights_structure['fc1']['weight']
    fc1_b = weights_structure['fc1']['bias']
    fc2_w = weights_structure['fc2']['weight']
    fc2_b = weights_structure['fc2']['bias']
    fc3_w = weights_structure['fc3']['weight']
    fc3_b = weights_structure['fc3']['bias']

    i = 0

    for item in range(len(fc1_w)):
        for num in range(len(fc1_w[item])):
            fc1_w[item][num] = int(dwload_weights[i]) / 10000
            i = i + 1
    for item in range(len(fc1_b)):
        fc1_b[item] = int(dwload_weights[i]) / 10000
        i = i + 1

    for item in range(len(fc2_w)):
        for num in range(len(fc2_w[item])):
            fc2_w[item][num] = int(dwload_weights[i]) / 10000
            i = i + 1
    for item in range(len(fc2_b)):
        fc2_b[item] = int(dwload_weights[i]) / 10000
        i = i + 1

    for item in range(len(fc3_w)):
        for num in range(len(fc3_w[item])):
            fc3_w[item][num] = int(dwload_weights[i]) / 10000
            i = i + 1
    for item in range(len(fc3_b)):
        fc3_b[item] = int(dwload_weights[i]) / 10000
        i = i + 1
    # print(fc1_w)
    # print("--" * 10)
    # print(fc1_b)
    # print("---" * 10)

    model.fc1[0].weight.data = torch.from_numpy(np.array(fc1_w, dtype=float32))
    model.fc1[0].bias.data = torch.from_numpy(np.array(fc1_b, dtype=float32))
    model.fc2[0].weight.data = torch.from_numpy(np.array(fc2_w, dtype=float32))
    model.fc2[0].bias.data = torch.from_numpy(np.array(fc2_b, dtype=float32))
    model.fc3[0].weight.data = torch.from_numpy(np.array(fc3_w, dtype=float32))
    model.fc3[0].bias.data = torch.from_numpy(np.array(fc3_b, dtype=float32))
    torch.save(model.state_dict(), './outputs/dw_weighted_model.pkl')


# load smart contract address


def load_contract(eth):
    with open("./outputs/contracts.json", 'r') as f:
        contracts = json.loads(f.read())
    mc = eth.contract(contracts['model.sol']
                      ['address'], contracts['model.sol']['abi'])
    return mc


# download model parameters from smart contract 左闭右开


def download_weights(start_index, size):
    updates = []
    print('[Info] Start downloading model parameters...')

    start = time.time()
    for i in range(size):
        updates.append(model_contract.functions.GetUpdate(
            start_index + i).call())
        if i == size - 1:
            print(
                '[Info] Parameter download progress: [{}/{}]'.format(i + 1, size), end='\n')
        else:
            print(
                '[Info] Parameter download progress: [{}/{}]'.format(i + 1, size), end='\r')

    end = time.time()
    print('[Info] Parameter download take %s s' % (end - start))
    with open('./outputs/download_weights.txt', 'w') as f:
        f.write(str(updates))
    return updates


def loadData(batch_size):
    print("Read the pickle file of test set...")
    with open('dataset/test_set.pickle', 'rb') as f:
        test_set = pickle.load(f)

    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True)

    print("test_size:", len(test_set))
    return test_loader


def testModel(model, test_loader, img_H, img_W, device):
    start = time.time()
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        images, labels = data['image'], data['label']
        images = images.reshape(-1, img_H * img_W * 3)
        # images = images.to(torch.float32)
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        values, predicte = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicte == labels).sum().item()
    print("The accuracy of total {} images: {:.4f}%".format(
        total, 100 * correct / total))
    end = time.time()
    print('[Info] Model test take %s s' % (end - start))


if __name__ == "__main__":
    args_par = arg_parser.parse_args()
    batch_size = args_par.BATCH_SIZE
    num_classes = args_par.NUM_CLASSES
    img_H = args_par.IMAGE_HEIGHT
    img_W = args_par.IMAGE_WIDTH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO]The device used for Testing Model is {}".format(device))
    if not os.path.exists("./outputs/download_weights.txt"):
        # 解析节点参数 上传节点默认为第一个节点
        # create eth interface 定义上传权重的节点 默认为第一个节点
        nodes = getNodes("./config/node_config.json")
        eth = EthInstance(nodes[args_par.NODE_NUM - 1].ip,
                          nodes[args_par.NODE_NUM - 1].port)

        # 第四个节点为矿工节点
        # miner_node = EthInstance(nodes[3].ip, nodes[3].port)

        # load smart contract
        model_contract = load_contract(eth)

        print("[INFO]The length of the model's weights is {} .".format(167060))
        # download model parameters from smart contract
        dwload_weights = download_weights(0, 167060)

    with open("./outputs/download_weights.txt", 'r') as f:
        dwload_weights = f.read()

    dwload_weights = dwload_weights[1:-1].split(',')
    # print(dwload_weights)

    input_num = img_H * img_W * 3
    hidden1_num = 64
    hidden2_num = 16
    output_num = num_classes
    model = NeuralNet(input_num, hidden1_num, hidden2_num, output_num)

    # load model weights length:167060
    loadWightsIntoLocalModel(model, "./weights/total.json", dwload_weights)
    model = model.to(device)

    test_data = loadData(batch_size)

    testModel(model, test_data, img_H, img_W, device)
