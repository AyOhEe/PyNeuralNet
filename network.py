import numpy as np
import copy
import struct

#TODO: add ranges to all of the node indexes. makes it clearer what the shit any of this expects in

#default activation function
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

#a network, containing all of its nodes and its connection matrix
class Network:
    #constructor
    def __init__(self, genome_path, inputs, outputs, activation_function=Sigmoid):
        #set up the input and output node arrays
        self.inputs = np.array([0 for _ in range(inputs)], dtype=np.float32)
        self.outputs = np.array([0 for _ in range(outputs)], dtype=np.float32)
        #store the activation function
        self.activate = activation_function

        #open and read the node chromosome
        node_dt = np.dtype("<f4")
        node_data = np.fromfile(genome_path + "/Nodes.chr", dtype = node_dt)
        #set up the nodes and biases
        self.nodes = np.array([0 for _ in range(len(node_data))], dtype=np.float32)
        self.biases = np.array([n[0] for n in node_data], dtype=np.float32)

        #open and read the connection chromosome
        conn_dt = np.dtype([("input", "<i2"), ("output", "<i2"), ("weight", "<f4")])
        conn_data = np.fromfile(genome_path + "/Connections.chr", dtype = conn_dt)
        self.connection_matrix = np.array([
            [0 for _ in range(len(self.nodes) + len(self.outputs))] #how the n'th input contributes to each output
            for _ in range(len(self.inputs) + len(self.nodes))      #where n is the row index
        ], dtype=np.float32)
        #fill the connection matrix
        for conn in conn_data:
            self.connection_matrix[conn[0]][conn[1]] = conn[2]


    #returns a copy of this network
    def copy(self):
        return copy.deepcopy(self)

    #returns the node index for the n'th connection into this node
    def ConnectionIndexToSourceIndex(self, NodeIndex, ConnectionIndex):
        #the connection between input 0 and output -1(final) will be at the end of the 1st row (0, -1)
        #the connection between input 0 and the first node will be at the start of the 1st row (0, 0)
        #the connection between two nodes is at (n1, n2), n1 < ins+nodes, n2 < nodes+outs

        #find the relevant column in the connection array
        weights = self.connection_matrix[:, NodeIndex]

        #iterate through the weights until the n'th non-zero weight is found and return the index
        count = 0
        for i in range(len(weights)):
            if weights[i] == 0:
                continue
            if count == ConnectionIndex:
                return i
            count += 1

        print(f"Could not find valid connection for o{NodeIndex}.c{ConnectionIndex}")
        return "FAIL"

    #looks at and returns the outputs from the last call of CalculateOutputs
    def PeekOutputs(self):
        return self.outputs

    #takes one step
    def _step_one(self):
        #get an array of all potential
        ins = np.hstack((self.inputs, self.nodes))
        #calculate the outputs
        outs = np.matmul(ins, self.connection_matrix) + np.hstack((self.biases, np.zeros(self.GetOutputCount())))
        #apply the activation function
        outs = np.vectorize(self.activate)(outs)

        #separate into internal node values and output values
        self.nodes = outs[0:-self.GetOutputCount()]
        self.outputs = outs[-self.GetOutputCount():]

    #calculates and returns the outputs of the network given inputs
    def Step(self, inputs, n_steps=1):
        #set the value of the input nodes to the given inputs
        self.inputs = np.array(inputs)

        #loop for the number of steps, and compute the outputs for each iteration
        for i in range(n_steps):
            self._step_one()

        #return the new outputs
        return self.PeekOutputs()

    #returns relevant information about the network
    def GetInputCount(self):
        #return the length of the input array
        return len(self.inputs)

    def GetNodeCount(self):
        #return the length of the node array
        return len(self.nodes)

    def GetOutputCount(self):
        #return the length of the output array
        return len(self.outputs)

    def GetConnectionCount(self):
        #return the number of connections in the network
        return np.count_nonzero(self.connection_matrix)

    def GetNetworkStructure(self):
        return (self.GetInputCount(), self.GetNodeCount(), self.GetOutputCount())

    #adds a node between a connection to the network
    def AddNode(self, NodeIndex, ConnectionIndex, bias):
        #find the target and source nodes
        source_index = self.ConnectionIndexToSourceIndex(NodeIndex, ConnectionIndex)
        target_index = NodeIndex

        #find the relevant connection (if it does not exist, simply exit out)
        if (old_weight := self.GetConnectionWeight(source_index, target_index)) == 0:
            print(f"Attempted to add node between i{source_index}->o{target_index}, but connection does not exist!")
            return
        #remove the connection
        self.SetConnectionWeight_st(source_index, target_index, 0)

        #make a new node, and make connections: old_source -(old_weight)> new_node -(1)> old_target
        self.nodes = np.append(self.nodes, np.array([0]))
        self.biases = np.append(self.biases, np.array([0]))
        self.connection_matrix = np.vstack((self.connection_matrix, np.zeros(self.GetNodeCount() - 1 + self.GetOutputCount())))
        self.connection_matrix = np.insert(
            self.connection_matrix,
            self.GetNodeCount() - 1,
            np.zeros(self.GetInputCount() + self.GetNodeCount()),
            axis=1
        )
        self.AddConnection(source_index, self.GetNodeCount() - 1, old_weight)
        self.AddConnection(self.GetInputCount() + self.GetNodeCount() - 1, target_index + 1, 1)

    #adds a connection to the network
    def AddConnection(self, SourceNodeIndex, TargetNodeIndex, weight):
        #set the corresponding value in the connection matrix to weight
        self.connection_matrix[SourceNodeIndex, TargetNodeIndex] = weight

    #removes a node from the network
    #NodeIndex: range from 0 - len(nodes)
    def RemoveNode(self, NodeIndex):
        #remove the corresponding row and column from the connection matrix
        self.connection_matrix = np.delete(self.connection_matrix, NodeIndex, 1)
        self.connection_matrix = np.delete(self.connection_matrix, NodeIndex + self.GetInputCount(), 0)

        #remove the node from the node array
        self.nodes = np.delete(self.nodes, NodeIndex, 0)
        self.biases = np.delete(self.biases, NodeIndex, 0)

    #removes a connection from the network
    def RemoveConnection(self, NodeIndex, ConnectionIndex):
        #set the weight of the connection to 0. thats as good as gone in a connection matrix
        self.SetConnectionWeight_st(self.ConnectionIndexToSourceIndex(NodeIndex, ConnectionIndex), NodeIndex, 0)

    #returns the bias of a node
    def GetNodeBias(self, NodeIndex):
        #find and return the relevant node bias
        return self.biases[NodeIndex]

    #sets the bias of a node
    def SetNodeBias(self, NodeIndex, bias):
        #find and set the relevant node bias
        self.biases[NodeIndex] += bias

    #returns the number of input connections this node has
    def GetNodeConnectionCount(self, NodeIndex):
        #find the relevant column in the connection array
        weights = self.connection_matrix[:, NodeIndex]

        #find the number of non-zero values it has
        return np.count_nonzero(weights)

    #returns the weight of a connection
    def GetConnectionWeight(self, SourceIndex, TargetIndex):
        #find and return the relevant weight in the connection matrix
        return self.connection_matrix[SourceIndex, TargetIndex]

    #sets the weight of a connection
    def SetConnectionWeight(self, NodeIndex, ConnectionIndex, weight):
        self.connection_matrix[self.ConnectionIndexToSourceIndex(NodeIndex, ConnectionIndex), NodeIndex] = weight

    #sets the weight of the connection between two nodes, where their indices are known
    def SetConnectionWeight_st(self, SourceIndex, TargetIndex, weight):
        self.connection_matrix[SourceIndex, TargetIndex] = weight


    #TODO
    #saves the network in it's current state to a file
    def SaveNetwork(self, GenomePath):
        #create the directory if it does not exist
        if not os.path.exists(GenomePath):
            os.makedirs(GenomePath)

        #open the node chromosome and write the biases
        node_chr = open(GenomePath + "/Nodes.chr", "wb")
        for b in self.biases:
            node_chr.write(struct.pack("<f", b))
        #close the node chromosome
        node_chr.close()

        #open the connection chromosome
        chromosome_chr = open(GenomePath + "/Connections.chr", "wb")
        #iterate through each connection and save it to the connection chromosome
        for source, target in np.ndindex(self.connection_matrix.shape):
            if not self.connection_matrix[source, target] == 0:
                chromosome_chr.write(struct.pack("<h", source))
                chromosome_chr.write(struct.pack("<h", target))
                chromosome_chr.write(struct.pack("<f", self.connection_matrix[source, target]))

        #close the connection chromosome
        chromosome_chr.close()

    def __str__(self):
        return str(self.GetNetworkStructure())

if __name__ == "__main__":
    #-----TEST 1-----
    print("-----TEST 1-----")
    #create a network
    net = Network(
        "Genomes/default", #genome path
        2,                                             #inputs
        1                                              #outputs
    )

    print(f"\n{net.GetNetworkStructure() = }")

    print("\nNodes: ")
    for i in range(net.GetNodeCount()):
        print(f"  node{i}_bias = {net.GetNodeBias(i)}")
        for j in range(net.GetNodeConnectionCount(i)):
            print(f"    node{i}_conn{j}_weight = {net.GetConnectionWeight(net.ConnectionIndexToSourceIndex(i, j), i)}")
    print("\nOutputNodes:")
    for i in range(net.GetNodeCount(), net.GetNodeCount() + net.GetOutputCount()):
        print(f"  output node{i}")
        for j in range(net.GetNodeConnectionCount(i)):
            print(f"    node{i}_conn{j}_weight = {net.GetConnectionWeight(net.ConnectionIndexToSourceIndex(i, j), i)}")

    print(f"\n{net.Step([0, 0], 2) = }")
    print(f"{net.Step([0, 1], 2) = }")
    print(f"{net.Step([1, 0], 2) = }")
    print(f"{net.Step([1, 1], 2) = }")

    net.AddNode(0, 0, -1.5)
    print("Added node")
    net.AddConnection(1, 0, 1.875)
    print("Added connection")
    net.SetNodeBias(0, 0.5)
    print("Set bias")
    net.SetConnectionWeight(0, 0, -0.875)
    print("Set weight")

    print(f"\n{net.GetNetworkStructure() = }")

    print("\nNodes: ")
    for i in range(net.GetNodeCount()):
        print(f"  node{i}_bias = {net.GetNodeBias(i)}")
        for j in range(net.GetNodeConnectionCount(i)):
            print(f"    node{i}_conn{j}_weight = {net.GetConnectionWeight(net.ConnectionIndexToSourceIndex(i, j), i)}")
    print("\nOutputNodes:")
    for i in range(net.GetNodeCount(), net.GetNodeCount() + net.GetOutputCount()):
        print(f"  output node{i}")
        for j in range(net.GetNodeConnectionCount(i)):
            print(f"    node{i}_conn{j}_weight = {net.GetConnectionWeight(net.ConnectionIndexToSourceIndex(i, j), i)}")

    print(f"\n{net.Step([0, 0], 2) = }")
    print(net.nodes)
    print(f"{net.Step([0, 1], 2) = }")
    print(net.nodes)
    print(f"{net.Step([1, 0], 2) = }")
    print(net.nodes)
    print(f"{net.Step([1, 1], 2) = }")
    print(net.nodes)

    print(net.connection_matrix)

    net.SaveNetwork("Genomes/altered_genome")

    #-----TEST 2-----
    print("\n\n-----TEST 2-----")
    #create a network
    net = Network(
        "Genomes/altered_genome", #genome path
        2,                                             #inputs
        1                                              #outputs
    )

    print(f"\n{net.GetNetworkStructure() = }")

    print("\nNodes: ")
    for i in range(net.GetNodeCount()):
        print(f"  node{i}_bias = {net.GetNodeBias(i)}")
        for j in range(net.GetNodeConnectionCount(i)):
            print(f"    node{i}_conn{j}_weight = {net.GetConnectionWeight(net.ConnectionIndexToSourceIndex(i, j), i)}")
    print("\nOutputNodes:")
    for i in range(net.GetNodeCount(), net.GetNodeCount() + net.GetOutputCount()):
        print(f"  output node{i}")
        for j in range(net.GetNodeConnectionCount(i)):
            print(f"    node{i}_conn{j}_weight = {net.GetConnectionWeight(net.ConnectionIndexToSourceIndex(i, j), i)}")

    print(f"\n{net.Step([0, 0], 2) = }")
    print(f"{net.Step([0, 1], 2) = }")
    print(f"{net.Step([1, 0], 2) = }")
    print(f"{net.Step([1, 1], 2) = }")

    net.RemoveNode(0)
    net.RemoveConnection(0, 0)

    print(f"\n{net.GetNetworkStructure() = }")

    print("\nNodes: ")
    for i in range(net.GetNodeCount()):
        print(f"  node{i}_bias = {net.GetNodeBias(i)}")
        for j in range(net.GetNodeConnectionCount(i)):
            print(f"    node{i}_conn{j}_weight = {net.GetConnectionWeight(net.ConnectionIndexToSourceIndex(i, j), i)}")
    print("\nOutputNodes:")
    for i in range(net.GetNodeCount(), net.GetNodeCount() + net.GetOutputCount()):
        print(f"  output node{i}")
        for j in range(net.GetNodeConnectionCount(i)):
            print(f"    node{i}_conn{j}_weight = {net.GetConnectionWeight(net.ConnectionIndexToSourceIndex(i, j), i)}")

    print(f"\n{net.Step([0, 0], 2) = }")
    print(f"{net.Step([0, 1], 2) = }")
    print(f"{net.Step([1, 0], 2) = }")
    print(f"{net.Step([1, 1], 2) = }")

    net.SaveNetwork("Genomes/further_altered_genome")
