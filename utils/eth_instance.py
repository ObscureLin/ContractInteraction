from web3 import Web3
import solcx
from solcx import compile_source


class EthInstance:

    def __init__(self, ip, port):
        self.w3 = Web3(Web3.HTTPProvider("http://" + str(ip) + ":" + str(port)))

    # Eth API
    def coinbase(self):
        return self.w3.eth.coinbase

    def mining(self):
        return self.w3.eth.mining

    def accounts(self):
        return self.w3.eth.accounts

    def blockNumber(self):
        return self.w3.eth.block_number

    def getBalance(self, addr):
        return self.w3.eth.get_balance(addr)

    def getCode(self, addr):
        return self.w3.eth.get_code(addr)

    def getBlock(self, num):
        return self.w3.eth.get_block(num)

    def getTransaction(self, hash):
        return self.w3.eth.get_transaction(hash)

    def sendTransaction(self, json):
        return self.w3.eth.send_transaction(json)

    def sendRawTransaction(self, rtx):
        return self.w3.eth.sendRawTransaction(rtx)

    def getReceipt(self, hash):
        return self.w3.eth.wait_for_transaction_receipt(hash)

    def estimateGas(self, tx):
        return self.w3.eth.estimateGas(tx)

    def signTx(self, tx, pk):
        return self.w3.eth.account.sign_transaction(tx, private_key=pk)

    # Miner API
    def minerStart(self):
        self.w3.geth.miner.start()

    def minerStop(self):
        self.w3.geth.miner.stop()

    # Geth API
    def nodeInfo(self):
        return self.w3.geth.admin.node_info()

    def peers(self):
        return self.w3.geth.admin.peers()

    def addPeer(self, enode):
        return self.w3.geth.admin.add_peer(enode)

    def newAccount(self, pwd):
        return self.w3.geth.personal.new_account(pwd)

    def unlockAccount(self, addr, pwd):
        return self.w3.geth.personal.unlock_account(addr, pwd)

    # Contract API
    # smart contract compilation
    def compile(self, path):
        with open(path, 'r') as f:
            sol = f.read()
        solcx.set_solc_version('0.7.0')
        compiled_sol = compile_source(sol)
        contract_id, contract_interface = compiled_sol.popitem()
        return contract_interface

    # smart contract deployment
    def deploy(self, addr, interface):
        self.w3.geth.personal.unlock_account(addr, "123")
        self.w3.eth.default_account = self.w3.eth.accounts[0]
        MyContract = self.w3.eth.contract(abi=interface['abi'], bytecode=interface['bin'])
        tx_hash = MyContract.constructor().transact()
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_receipt

    # smart contract execution
    def contract(self, addr, abi):
        MyContract = self.w3.eth.contract(address=addr, abi=abi)
        return MyContract
