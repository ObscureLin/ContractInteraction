pragma solidity >=0.7.0 <0.9.0;

contract Model {
    // latest global model parameters
    int256[167060] weight;
    // Model parameter updates received in a round
    int256[167060] update;
    // Number of local updates received in a round
    int8 count = 0;
    // Model parameter version
    int8 version = 0;
    // Minimum number of nodes required for each update round
    int8 min = 3;

    // Receive local updates uploaded by the nodes
    function SetUpdate(uint32 loc, int256 v) public {
        update[loc] = v;
    }

    // Receive local updates uploaded by the nodes
    function SetUpdates(uint32 index, uint8 start, int256[240] memory vs) public {
        for (uint8 i = 0; i < 240 - start; i++) {
            update[index + i] = vs[start + i];
        }
    }

    // Update global parameters after collecting enough local updates
    function UpdateWeights() private {
        for (uint32 i = 0; i < update.length; i++)
            weight[i] = update[i] / count;
        version++;
        count = 0;
    }

    function GetWeights() public view returns (int256[167060] memory) {
        return weight;
    }

    function GetWeight(uint32 loc) public view returns (int256) {
        return weight[loc];
    }

    function GetUpdates() public view returns (int256[167060] memory) {
        return update;
    }

    function GetUpdate(uint32 loc) public view returns (int256) {
        return update[loc];
    }

    function GetCount() public view returns (int8) {
        return count;
    }

    function GetVersion() public view returns (int8) {
        return version;
    }

    function GetMin() public view returns (int8) {
        return min;
    }

    function SetMin(int8 m) public {
        min = m;
    }
}
