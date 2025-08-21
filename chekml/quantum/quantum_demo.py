import torch
from quantum_hybrid import QuantumHybridModel

def demo_quantum_hybrid():
    """Demonstrate the QuantumHybridModel for regression."""
    # Set parameters
    num_qubits = 4
    num_train_samples = 100
    num_test_samples = 20

    # Generate synthetic data
    x_train = (torch.rand((num_train_samples, num_qubits)) - 0.5) * 2  # Scale to [-1,1]
    y_train = torch.sin(x_train.sum(dim=1)).reshape(-1, 1)
    
    x_test = (torch.rand((num_test_samples, num_qubits)) - 0.5) * 2
    y_test = torch.sin(x_test.sum(dim=1)).reshape(-1, 1)

    # Initialize and train model
    model = QuantumHybridModel(num_qubits=num_qubits)
    print("Training Quantum Hybrid Model...")
    model.train_model(x_train, y_train, num_epochs=20, lr=0.001)

    # Evaluate model
    print("\nEvaluating Model...")
    metrics = model.evaluate(x_test, y_test)
    
    print("\n📊 Model Evaluation Results:")
    print(f"🔹 Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"🔹 Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"🔹 R² Score: {metrics['r2']:.4f}")

if __name__ == "__main__":
    demo_quantum_hybrid()
