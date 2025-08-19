import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class QuantumHybridModel(nn.Module):
    """Hybrid Quantum-Classical Neural Network for regression."""
    
    def __init__(self, num_qubits=4, confidence_threshold=0.8):
        """Initialize the hybrid quantum-classical model.
        
        Args:
            num_qubits (int): Number of qubits for the quantum circuit
            confidence_threshold (float): Threshold for confidence (not used in regression, kept for compatibility)
        """
        super(QuantumHybridModel, self).__init__()
        self.num_qubits = num_qubits
        self.confidence_threshold = confidence_threshold
        
        # Create quantum circuit
        self.qc, self.input_params, self.weight_params = self._create_quantum_circuit()
        
        # Create QNN
        self.qnn = self._create_qnn()
        
        # Classical layers
        self.fc1 = nn.Linear(self.num_qubits, 4)  # Hidden layer
        self.fc2 = nn.Linear(4, 1)  # Output layer
        self.torch_qnn = TorchConnector(self.qnn)

    def _create_quantum_circuit(self):
        """Create the quantum circuit for the QNN.
        
        Returns:
            tuple: (QuantumCircuit, input_params, weight_params)
        """
        qc = QuantumCircuit(self.num_qubits)
        input_params = ParameterVector("x", length=self.num_qubits)
        weight_params = ParameterVector("theta", length=self.num_qubits * 3)

        # Apply rotation layers
        for i in range(self.num_qubits):
            qc.ry(input_params[i], i)
            qc.rx(weight_params[i], i)

        qc.barrier()

        for i in range(self.num_qubits):
            qc.ry(weight_params[self.num_qubits + i], i)
            qc.rx(weight_params[2 * self.num_qubits + i], i)

        # Add entanglement
        for i in range(self.num_qubits):
            qc.cx(i, (i + 1) % self.num_qubits)
            qc.cx((i + 1) % self.num_qubits, i)

        return qc, input_params, weight_params

    def _create_qnn(self):
        """Create the SamplerQNN for the quantum circuit.
        
        Returns:
            SamplerQNN: Configured quantum neural network
        """
        def interpret(bitstring):
            if isinstance(bitstring, int):
                return bitstring % self.num_qubits
            elif isinstance(bitstring, (list, np.ndarray)):
                return sum(int(b) for b in bitstring) % self.num_qubits
            else:
                raise ValueError(f"Unexpected bitstring format: {bitstring}")

        return SamplerQNN(
            circuit=self.qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sparse=False,
            interpret=interpret,
            output_shape=self.num_qubits
        )

    def forward(self, x):
        """Forward pass through the hybrid model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_qubits)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        q_out = self.torch_qnn(x)
        q_out = torch.relu(self.fc1(q_out))
        return self.fc2(q_out)

    def train_model(self, x_train, y_train, num_epochs=150, lr=0.001):
        """Train the hybrid model.
        
        Args:
            x_train (torch.Tensor): Training input data
            y_train (torch.Tensor): Training target data
            num_epochs (int): Number of training epochs
            lr (float): Learning rate
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            predictions = self(x_train)
            loss = loss_fn(predictions, y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:  # Print every 10 epochs
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

    def evaluate(self, x_test, y_test):
        """Evaluate the model on test data.
        
        Args:
            x_test (torch.Tensor): Test input data
            y_test (torch.Tensor): Test target data
            
        Returns:
            dict: Dictionary containing MSE, MAE, and R2 scores
        """
        self.eval()
        with torch.no_grad():
            predictions = self(x_test).detach().numpy()
        y_test_np = y_test.numpy()

        return {
            "mse": mean_squared_error(y_test_np, predictions),
            "mae": mean_absolute_error(y_test_np, predictions),
            "r2": r2_score(y_test_np, predictions)
        }

if __name__ == "__main__":
    print("This is a module. Please use demo.py to run the QuantumHybridModel.")
