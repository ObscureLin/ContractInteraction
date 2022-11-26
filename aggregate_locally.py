# 该文件代码用于在本地验证聚合node1 2 3的权重后的模型的精度
from dwload_verify import *
import torch
import json
from utils import arg_parser


def load_weights(weights_path):
    with open(weights_path, 'r') as f:
        model = json.load(f)
    weights = []
    fc1_w = model['fc1']['weight']
    fc1_b = model['fc1']['bias']
    fc2_w = model['fc2']['weight']
    fc2_b = model['fc2']['bias']
    fc3_w = model['fc3']['weight']
    fc3_b = model['fc3']['bias']

    for item in fc1_w:
        for num in item:
            weights.append(int(num))
    for item in fc1_b:
        weights.append(int(item))
    for item in fc2_w:
        for num in item:
            weights.append(int(num))
    for item in fc2_b:
        weights.append(int(item))
    for item in fc3_w:
        for num in item:
            weights.append(int(num))
    for item in fc3_b:
        weights.append(int(item))
    return weights


if __name__ == "__main__":
    args_par = arg_parser.parse_args()
    batch_size = args_par.BATCH_SIZE
    num_classes = args_par.NUM_CLASSES
    img_H = args_par.IMAGE_HEIGHT
    img_W = args_par.IMAGE_WIDTH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO]The device used for Testing Model is {}".format(device))

    node1_weights = load_weights("weights/node1.json")
    node2_weights = load_weights("weights/node2.json")
    node3_weights = load_weights("weights/node3.json")

    aggregated_weights = []
    for index in range(len(node1_weights)):
        aggregated_weights.append((node1_weights[index] + node2_weights[index] + node3_weights[index]) / 3)
        if index == 0 or index == 1:
            print(
                "[INFO] n1: {} n2: {} n3: {} agg: {}".format(node1_weights[index], node2_weights[index],
                                                             node3_weights[index],
                                                             aggregated_weights[index]))

    # 初始化本地深度学习模型
    input_num = img_H * img_W * 3
    hidden1_num = 64
    hidden2_num = 16
    output_num = num_classes
    model = NeuralNet(input_num, hidden1_num, hidden2_num, output_num)
    loadWightsIntoLocalModel(model, "./weights/total.json", aggregated_weights)
    model = model.to(device)

    # 加载测试集,测试模型
    test_data = loadData(batch_size)
    testModel(model, test_data, img_H, img_W, device)
