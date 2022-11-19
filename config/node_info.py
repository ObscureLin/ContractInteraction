import json


class Node:
    def __init__(self, no, ip, port):
        self.no = no
        self.ip = ip
        self.port = port


def getNodes(cfg_path):
    with open(cfg_path, 'r') as file:
        data = json.load(file)

    count = data['nodecount']
    ip = data['startip']
    port = data['startport']
    words = ip.split('.')

    nodes = []
    for i in range(count):
        nodes.append(Node(i + 1, words[0] + "." + words[1] + "." + words[2] + "." + str(int(words[-1]) + i), port + i))
    return nodes
