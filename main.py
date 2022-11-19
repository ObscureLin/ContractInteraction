from config.node_info import *
from utils.eth_instance import EthInstance
from utils.arg_parser import *
import json
import time


# load model weights
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


# load smart contract address
def load_contract(eth):
    with open("./outputs/contracts.json", 'r') as f:
        contracts = json.loads(f.read())
    mc = eth.contract(contracts['model.sol']['address'], contracts['model.sol']['abi'])
    return mc


# upload model parameters to smart contract
def upload_weights(eth, weights, num):
    size = len(weights)
    batch = 240
    gas = []
    cnt = 0

    eth.minerStart()
    print('[Info] Start uploading model parameters...')

    start = time.time()
    for i in range(0, num, batch):
        if ((cnt / batch) % 100 == 0):
            eth.unlockAccount(eth.accounts()[0], "123")
        cnt = i + batch
        if (i + batch > size):
            cnt = size
            tx_hash = model_contract.functions.SetUpdates(i, i + batch - size, weights[-1 * batch:]).transact(
                {'from': eth.accounts()[0]})
            tx_receipt = eth.getReceipt(tx_hash)
            gas.append(tx_receipt['gasUsed'])
            print('[Info] Parameter upload progress: [{}/{}]'.format(cnt, size), end="\n")
        else:
            tx_hash = model_contract.functions.SetUpdates(i, 0, weights[i: i + batch]).transact(
                {'from': eth.accounts()[0]})
            tx_receipt = eth.getReceipt(tx_hash)
            gas.append(tx_receipt['gasUsed'])
            print('[Info] Parameter upload progress: [{}/{}]'.format(cnt, size), end="\r")

    end = time.time()

    print('[Info] Parameter upload take %s s' % (end - start))
    eth.minerStop()
    print('[Info] Gas used: ', sum(gas))
    print('[Info] Gas used of each tx: ', gas)


# download model parameters from smart contract
def download_weights(start_index, size):
    updates = []
    print('[Info] Start downloading model parameters...')

    start = time.time()
    for i in range(size):
        updates.append(model_contract.functions.GetUpdate(start_index + i).call())
        if i == size - 1:
            print('[Info] Parameter download progress: [{}/{}]'.format(i + 1, size), end='\n')
        else:
            print('[Info] Parameter download progress: [{}/{}]'.format(i + 1, size), end='\r')

    end = time.time()
    print('[Info] Parameter download take %s s' % (end - start))
    return updates


if __name__ == "__main__":
    # 解析节点参数 默认为第一个节点
    args_par = parse_args()

    # load model weights length:167060
    weights = load_weights("./weights/total.json")

    # create eth interface 定义上传权重的节点 默认为第一个节点
    nodes = getNodes("./config/node_config.json")
    eth = EthInstance(nodes[args_par.NODE_NUM - 1].ip, nodes[args_par.NODE_NUM - 1].port)

    # load smart contract
    model_contract = load_contract(eth)

    # upload model parameters to smart contract 权重总长度
    upload_weights(eth, weights, len(weights))

    # download model parameters from smart contract
    # updates = download_weights(0, len(weights))
