from utils.eth_instance import EthInstance
from config.node_info import *
from hexbytes import HexBytes
import json


class HexJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, HexBytes):
            return obj.hex()
        return super().default(obj)


if __name__ == "__main__":
    # create eth interface
    nodes = getNodes("./config/node_config.json")
    eth = EthInstance(nodes[3].ip, nodes[3].port)  # 默认四号节点部署合约

    # smart contract compilation and deployment
    eth.minerStart()
    print('[Info] Loading contract files from contracts/model.sol')
    contract_interface = eth.compile('./contracts/model.sol')
    tx_receipt = eth.deploy(eth.accounts()[0], contract_interface)
    print('[Info] The contract has been deployed successfully at address:', tx_receipt['contractAddress'])
    eth.minerStop()

    # save smart contract information
    model_contract = {"name": "model", "address": tx_receipt['contractAddress'], "abi": contract_interface['abi'],
                      "bytecode": contract_interface['bin'],
                      "receipt": json.dumps(dict(tx_receipt), cls=HexJsonEncoder)}
    contracts = {"model.sol": model_contract}
    with open('./outputs/contracts.json', 'w') as f:
        f.write(json.dumps(contracts))
    print('[Info] The contract information has been written in ./outputs/contracts.json')
