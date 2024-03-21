import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
# from qiskit.circuit.library import NeuralNetworkCircuit
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC

class QuantumGatingFunction(nn.Module):
    def __init__(self, input_size, output_size, num_qubits):
        super(QuantumGatingFunction, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.num_qubits = num_qubits

    def forward(self, x):
        gating_weights = F.softmax(self.fc1(x), dim=1)
        qc = QuantumCircuit(self.num_qubits)
        qc = VQC(gating_weights, qc)
        return qc

class QuantumExpertNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_qubits):
        super(QuantumExpertNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.num_qubits = num_qubits

    def forward(self, x):
        expert_weights = self.fc1(x)
        qc = QuantumCircuit(self.num_qubits)
        qc = VQC(expert_weights, qc)
        return qc

class QuantumSoftDecisionTree(nn.Module):
    def __init__(self, input_size, output_size, num_nodes, num_experts, num_qubits):
        super(QuantumSoftDecisionTree, self).__init__()
        self.num_nodes = num_nodes
        self.num_experts = num_experts
        self.num_qubits = num_qubits
        self.gating_functions = nn.ModuleList([QuantumGatingFunction(input_size, num_experts, num_qubits) for _ in range(num_nodes)])
        self.expert_networks = nn.ModuleList([QuantumExpertNetwork(input_size, output_size, num_qubits) for _ in range(num_experts)])

    def forward(self, x):
        outputs = []
        gating_weights = [torch.ones(x.size(0), self.num_experts) / self.num_experts]  # Initialize with uniform weights

        for i in range(self.num_nodes):
            expert_outputs = [expert(x) for expert in self.expert_networks]
            weighted_outputs = [gating_weights[i][:, j].unsqueeze(1) * expert_outputs[j] for j in range(self.num_experts)]
            output = sum(weighted_outputs)
            outputs.append(output)
            gating_weights.append(self.gating_functions[i](x))

        return outputs[-1]