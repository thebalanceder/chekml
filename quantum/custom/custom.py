import numpy as np
import gym
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

# Define a Quantum Neural Network (QNN)
def create_quantum_neural_network(num_qubits):
    qc = QuantumCircuit(num_qubits)
    input_params = ParameterVector("x", length=num_qubits)
    weight_params = ParameterVector("theta", length=num_qubits * 3)  # More parameters

    # Apply three layers of Ry and Rx rotations
    for i in range(num_qubits):
        qc.ry(input_params[i], i)
        qc.rx(weight_params[i], i)

    qc.barrier()

    for i in range(num_qubits):
        qc.ry(weight_params[num_qubits + i], i)
        qc.rx(weight_params[2 * num_qubits + i], i)  # Extra Rx layer

    # Add stronger entanglement
    for i in range(num_qubits):
        qc.cx(i, (i + 1) % num_qubits)
        qc.cx((i + 1) % num_qubits, i)

    return qc, input_params, weight_params

# Create Quantum Neural Network
num_qubits = 4
quantum_circuit, input_params, weight_params = create_quantum_neural_network(num_qubits)
print(quantum_circuit.draw(output='text'))

# Define an interpretation function to reduce output size
def interpret(bitstring):
    #Ensure the output is within the valid range of `num_qubits`.
    if isinstance(bitstring, int):
        return bitstring % num_qubits  # Ensure index is between 0 and num_qubits-1
    elif isinstance(bitstring, (list, np.ndarray)):
        return sum(int(b) for b in bitstring) % num_qubits  # Sum bits and mod
    else:
        raise ValueError(f"Unexpected bitstring format: {bitstring}")

qnn = SamplerQNN(
    circuit=quantum_circuit,
    input_params=input_params,
    weight_params=weight_params,
    sparse=False,
    interpret=interpret,
    output_shape=num_qubits  # Ensures correct output size
)

# PyTorch Connector for Hybrid Quantum-Classical Model
class HybridQuantumModel(nn.Module):
    def __init__(self, qnn, num_qubits):
        super(HybridQuantumModel, self).__init__()
        self.qnn = TorchConnector(qnn)
        self.fc1 = nn.Linear(num_qubits, 4)  # Hidden layer (4 → 8 neurons)
        self.fc2 = nn.Linear(4, 1)  # Output layer
    
    def forward(self, x):
        q_out = self.qnn(x)
        print("Quantum Output Shape:", q_out.shape)
        q_out = torch.relu(self.fc1(q_out))  # Apply activation function
        return self.fc2(q_out)  # Output layer

# Define Hybrid Model
model = HybridQuantumModel(qnn, num_qubits)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Reduce LR
loss_fn = nn.MSELoss()

# Generate Training Data
x_train = (torch.rand((100, num_qubits)) - 0.5) * 2  # Scale to [-1,1]
y_train = torch.sin(x_train.sum(dim=1)).reshape(-1, 1)

# Train the Model
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(x_train)
    loss = loss_fn(predictions, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Evaluate the Model
x_test = torch.rand((10, num_qubits))
y_test = torch.sin(x_test.sum(dim=1)).reshape(-1, 1)
predictions = model(x_test)
print("Test Predictions:", predictions.detach().numpy())
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Generate test data
x_test = torch.rand((20, num_qubits))  # 20 test samples
y_test = torch.sin(x_test.sum(dim=1)).reshape(-1, 1)  # True outputs

# Make predictions
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    predictions = model(x_test).detach().numpy()

# Convert to NumPy for metric calculations
y_test_np = y_test.numpy()

# Compute evaluation metrics
mse = mean_squared_error(y_test_np, predictions)
mae = mean_absolute_error(y_test_np, predictions)
r2 = r2_score(y_test_np, predictions)

# Print results
print(f"📊 Model Evaluation:")
print(f"🔹 Mean Squared Error (MSE): {mse:.4f}")
print(f"🔹 Mean Absolute Error (MAE): {mae:.4f}")
print(f"🔹 R² Score: {r2:.4f}")
