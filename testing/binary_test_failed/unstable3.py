from collections import defaultdict
import random
import math
import numpy as np

# Initialize a logic table as a NumPy array
def initialize_logic_table():
    # Shape: (2,2,2,2) for A,B,C,(Output1_P1, Output2_P1)
    table = np.full((2, 2, 2, 2), 0.5, dtype=np.float32)
    return table

# Training function with advanced optimizations
def train(training_sets, epochs=300, base_lr_min=0.05, base_lr_max=0.2, batch_size=32, perturbation_rate=0.01, regularization=0.001):
    # Determine bits needed
    max_decimal = max(max(abs(int(a)), abs(int(b)), abs(int(out))) for a, b, out in training_sets)
    bits = math.ceil(math.log2(max_decimal + 1)) + 1
    
    # Initialize logic tables as NumPy arrays
    logic_tables = [initialize_logic_table() for _ in range(bits)]
    final_logic_table = initialize_logic_table()
    
    # Gradient moving averages for RMSProp-like updates
    grad_squares = [np.zeros_like(table) for table in logic_tables]
    final_grad_squares = np.zeros_like(final_logic_table)
    
    # Validation set
    validation_sets = training_sets[::5]
    training_sets = [t for t in training_sets if t not in validation_sets]
    
    best_val_error = float('inf')
    best_logic_tables = logic_tables
    best_final_table = final_logic_table
    patience = 50
    patience_counter = 0
    
    for epoch in range(epochs):
        total_train_hamming = 0
        total_train_decimal = 0
        random.shuffle(training_sets)
        
        # Cyclic learning rate (triangular policy)
        cycle = epoch % 20
        lr = base_lr_min + (base_lr_max - base_lr_min) * (1 - abs((cycle - 10) / 10))
        
        for batch_start in range(0, len(training_sets), batch_size):
            batch = training_sets[batch_start:batch_start + batch_size]
            
            for decimal_a, decimal_b, expected_output in batch:
                bin_a = format(int(decimal_a), '0' + str(bits) + 'b')
                bin_b = format(int(decimal_b), '0' + str(bits) + 'b')
                bin_expected = format(max(0, int(expected_output)), '0' + str(bits+1) + 'b')
                
                # Expected hidden binaries and carries
                expected_hidden = []
                expected_carries = [[] for _ in range(bits)]
                for digit_b_idx in range(bits-1, -1, -1):
                    b_bit = int(bin_b[digit_b_idx])
                    hidden = []
                    carry = 0
                    for digit_a_idx in range(bits-1, -1, -1):
                        a_bit = int(bin_a[digit_a_idx])
                        product = a_bit * b_bit
                        sum_bit = product + carry
                        hidden.append(str(sum_bit % 2))
                        carry = sum_bit // 2
                        expected_carries[bits-1-digit_b_idx].append(str(carry))
                    hidden = hidden[::-1]
                    shift = bits - 1 - digit_b_idx
                    if shift > 0:
                        hidden = ['0'] * shift + hidden[:-shift]
                    expected_hidden.append(hidden)
                
                # Simulate prediction
                hidden_binaries = []
                used_entries = [[] for _ in range(bits)]  # (a_idx, b_idx, c_idx, out1, out2, pos)
                final_used_entries = []
                
                for digit_b_idx in range(bits-1, -1, -1):
                    b_bit = bin_b[digit_b_idx]
                    b_idx = int(b_bit)
                    current_table = logic_tables[bits-1-digit_b_idx]
                    hidden_binary = []
                    carry = '0'
                    c_idx = 0
                    for digit_a_idx in range(bits-1, -1, -1):
                        a_bit = bin_a[digit_a_idx]
                        a_idx = int(a_bit)
                        c_idx = int(carry)
                        probs = current_table[a_idx, b_idx, c_idx]
                        out1 = '1' if random.random() < probs[0] else '0'
                        out2 = '1' if random.random() < probs[1] else '0'
                        hidden_binary.append(out2)
                        carry = out1
                        pos = bits - 1 - digit_a_idx
                        used_entries[bits-1-digit_b_idx].append((a_idx, b_idx, c_idx, out1, out2, pos))
                    hidden_binary = hidden_binary[::-1]
                    if (bits-1-digit_b_idx) > 0:
                        shift = bits-1-digit_b_idx
                        hidden_binary = ['0'] * shift + hidden_binary[:-shift]
                    hidden_binaries.append(hidden_binary)
                
                # Final layer
                result = []
                carry = '0'
                c_idx = 0
                for pos in range(bits):
                    bits_at_pos = [hb[pos] for hb in hidden_binaries]
                    a_bit = bits_at_pos[0] if len(bits_at_pos) > 0 else '0'
                    b_bit = bits_at_pos[1] if len(bits_at_pos) > 1 else '0'
                    a_idx = int(a_bit)
                    b_idx = int(b_bit)
                    c_idx = int(carry)
                    probs = final_logic_table[a_idx, b_idx, c_idx]
                    out1 = '1' if random.random() < probs[0] else '0'
                    out2 = '1' if random.random() < probs[1] else '0'
                    result.append(out2)
                    carry = out1
                    final_used_entries.append((a_idx, b_idx, c_idx, out1, out2, pos))
                a_idx = 0
                b_idx = 0
                c_idx = int(carry)
                probs = final_logic_table[a_idx, b_idx, c_idx]
                out1 = '1' if random.random() < probs[0] else '0'
                out2 = '1' if random.random() < probs[1] else '0'
                result.append(out2)
                final_used_entries.append((a_idx, b_idx, c_idx, out1, out2, bits))
                result = result[::-1]
                
                # Compute errors
                hamming_error = sum(a != b for a, b in zip(''.join(result), bin_expected))
                predicted_decimal = int(''.join(result), 2)
                decimal_error = abs(predicted_decimal - int(expected_output))
                total_train_hamming += hamming_error
                total_train_decimal += decimal_error
                
                # Update probabilities
                for table_idx, entries in enumerate(used_entries):
                    for a_idx, b_idx, c_idx, out1, out2, pos in entries:
                        # Output2
                        correct_out2 = bin_expected[pos] if pos < len(bin_expected) else '0'
                        delta = 1.0 if out2 == correct_out2 else -1.0
                        weight = 2 ** (pos / bits)  # Prioritize higher bits
                        grad = delta * weight
                        grad_squares[table_idx][a_idx, b_idx, c_idx, 1] = 0.9 * grad_squares[table_idx][a_idx, b_idx, c_idx, 1] + 0.1 * grad**2
                        rms = np.sqrt(grad_squares[table_idx][a_idx, b_idx, c_idx, 1] + 1e-8)
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 1] += lr * grad / rms
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 1] = max(0.01, min(0.99, logic_tables[table_idx][a_idx, b_idx, c_idx, 1]))
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 0] = max(0.01, min(0.99, logic_tables[table_idx][a_idx, b_idx, c_idx, 0]))
                        
                        # Output1
                        correct_out1 = expected_carries[table_idx][pos] if pos < len(expected_carries[table_idx]) else '0'
                        delta = 1.0 if out1 == correct_out1 else -1.0
                        grad = delta * weight * 2.0  # Higher weight for carries
                        grad_squares[table_idx][a_idx, b_idx, c_idx, 0] = 0.9 * grad_squares[table_idx][a_idx, b_idx, c_idx, 0] + 0.1 * grad**2
                        rms = np.sqrt(grad_squares[table_idx][a_idx, b_idx, c_idx, 0] + 1e-8)
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 0] += lr * grad / rms
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 0] = max(0.01, min(0.99, logic_tables[table_idx][a_idx, b_idx, c_idx, 0]))
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 0] = max(0.01, min(0.99, logic_tables[table_idx][a_idx, b_idx, c_idx, 0]))
                
                for a_idx, b_idx, c_idx, out1, out2, pos in final_used_entries:
                    correct_out2 = bin_expected[pos] if pos < len(bin_expected) else '0'
                    delta = 1.0 if out2 == correct_out2 else -1.0
                    weight = 2 ** (pos / bits)
                    grad = delta * weight
                    final_grad_squares[a_idx, b_idx, c_idx, 1] = 0.9 * final_grad_squares[a_idx, b_idx, c_idx, 1] + 0.1 * grad**2
                    rms = np.sqrt(final_grad_squares[a_idx, b_idx, c_idx, 1] + 1e-8)
                    final_logic_table[a_idx, b_idx, c_idx, 1] += lr * grad / rms
                    final_logic_table[a_idx, b_idx, c_idx, 1] = max(0.01, min(0.99, final_logic_table[a_idx, b_idx, c_idx, 1]))
                    final_logic_table[a_idx, b_idx, c_idx, 0] = max(0.01, min(0.99, final_logic_table[a_idx, b_idx, c_idx, 0]))
                
                # Regularization
                for table in logic_tables:
                    table[:,:,:,0] -= regularization * (table[:,:,:,0] - 0.5)
                    table[:,:,:,1] -= regularization * (table[:,:,:,1] - 0.5)
                    table[:,:,:,[0,1]] = np.clip(table[:,:,:,[0,1]], 0.01, 0.99)
                final_logic_table[:,:,:,0] -= regularization * (final_logic_table[:,:,:,0] - 0.5)
                final_logic_table[:,:,:,1] -= regularization * (final_logic_table[:,:,:,1] - 0.5)
                final_logic_table[:,:,:,[0,1]] = np.clip(final_logic_table[:,:,:,[0,1]], 0.01, 0.99)
                
                # Perturbation
                if random.random() < perturbation_rate:
                    for table in logic_tables:
                        noise = np.random.uniform(-0.05, 0.05, table.shape)
                        table += noise
                        table = np.clip(table, 0.01, 0.99)
                    noise = np.random.uniform(-0.05, 0.05, final_logic_table.shape)
                    final_logic_table += noise
                    final_logic_table = np.clip(final_logic_table, 0.01, 0.99)
        
        # Validation
        total_val_hamming = 0
        total_val_decimal = 0
        for decimal_a, decimal_b, expected_output in validation_sets:
            bin_expected = format(max(0, int(expected_output)), '0' + str(bits+1) + 'b')
            result = predict(decimal_a, decimal_b, logic_tables, final_logic_table, bits, firm=True, silent=True)
            result_bin = format(result, '0' + str(bits+1) + 'b')
            hamming_error = sum(a != b for a, b in zip(result_bin, bin_expected))
            decimal_error = abs(result - int(expected_output))
            total_val_hamming += hamming_error
            total_val_decimal += decimal_error
        
        avg_train_hamming = total_train_hamming / len(training_sets)
        avg_train_decimal = total_train_decimal / len(training_sets)
        avg_val_hamming = total_val_hamming / len(validation_sets)
        avg_val_decimal = total_val_decimal / len(validation_sets)
        loss = avg_val_hamming + 0.1 * avg_val_decimal
        
        # Early stopping
        if loss < best_val_error:
            best_val_error = loss
            best_logic_tables = [table.copy() for table in logic_tables]
            best_final_table = final_logic_table.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Epoch {epoch+1}, Train Hamming: {avg_train_hamming:.4f}, Train Decimal: {avg_train_decimal:.4f}, Val Hamming: {avg_val_hamming:.4f}, Val Decimal: {avg_val_decimal:.4f}, Loss: {loss:.4f}")
    
    return best_logic_tables, best_final_table, bits

# Prediction function (unchanged)
def predict(decimal_a, decimal_b, logic_tables, final_logic_table, bits, firm=False, silent=False):
    bin_a = format(int(decimal_a), '0' + str(bits) + 'b')
    bin_b = format(int(decimal_b), '0' + str(bits) + 'b')
    
    hidden_binaries = []
    
    for digit_b_idx in range(bits-1, -1, -1):
        b_bit = bin_b[digit_b_idx]
        b_idx = int(b_bit)
        current_logic_table = logic_tables[bits-1-digit_b_idx]
        hidden_binary = []
        carry = '0'
        c_idx = 0
        
        for digit_a_idx in range(bits-1, -1, -1):
            a_bit = bin_a[digit_a_idx]
            a_idx = int(a_bit)
            c_idx = int(carry)
            probs = current_logic_table[a_idx, b_idx, c_idx]
            
            if firm:
                out1 = '1' if probs[0] > 0.5 else '0'
                out2 = '1' if probs[1] > 0.5 else '0'
            else:
                out1 = '1' if random.random() < probs[0] else '0'
                out2 = '1' if random.random() < probs[1] else '0'
            
            hidden_binary.append(out2)
            carry = out1
            if not silent:
                print(f"Binary2 Digit {bits-digit_b_idx}: A={a_bit}, B={b_bit}, C={carry}, Output1={out1} (P1={probs[0]:.2f}), Output2={out2} (P1={probs[1]:.2f})")
        
        hidden_binary = hidden_binary[::-1]
        shift_amount = bits - 1 - digit_b_idx
        if shift_amount > 0:
            hidden_binary = ['0'] * shift_amount + hidden_binary[:-shift_amount]
        hidden_binaries.append(hidden_binary)
        if not silent:
            print(f"Hidden Binary {bits-digit_b_idx}: {hidden_binary}")
    
    result = []
    carry = '0'
    c_idx = 0
    for pos in range(bits):
        bits_at_pos = [hb[pos] for hb in hidden_binaries]
        a_bit = bits_at_pos[0] if len(bits_at_pos) > 0 else '0'
        b_bit = bits_at_pos[1] if len(bits_at_pos) > 1 else '0'
        a_idx = int(a_bit)
        b_idx = int(b_bit)
        c_idx = int(carry)
        probs = final_logic_table[a_idx, b_idx, c_idx]
        
        if firm:
            out1 = '1' if probs[0] > 0.5 else '0'
            out2 = '1' if probs[1] > 0.5 else '0'
        else:
            out1 = '1' if random.random() < probs[0] else '0'
            out2 = '1' if random.random() < probs[1] else '0'
        
        result.append(out2)
        carry = out1
        if not silent:
            print(f"Final Layer Pos {pos+1}: A={a_bit}, B={b_bit}, C={carry}, Output1={out1} (P1={probs[0]:.2f}), Output2={out2} (P1={probs[1]:.2f})")
    
    a_idx = 0
    b_idx = 0
    c_idx = int(carry)
    probs = final_logic_table[a_idx, b_idx, c_idx]
    if firm:
        out1 = '1' if probs[0] > 0.5 else '0'
        out2 = '1' if probs[1] > 0.5 else '0'
    else:
        out1 = '1' if random.random() < probs[0] else '0'
        out2 = '1' if random.random() < probs[1] else '0'
    result.append(out2)
    if not silent:
        print(f"Final Layer Extra Digit: A={a_bit}, B={b_bit}, C={out1}, Output1={out1} (P1={probs[0]:.2f}), Output2={out2} (P1={probs[1]:.2f})")
    
    result = result[::-1]
    if not silent:
        print(f"Result bits (MSB to LSB): {result}")
        binary_str = ''.join(result)
        print(f"Binary string: {binary_str}")
    decimal_output = int(''.join(result), 2)
    
    return decimal_output

# Example usage
if __name__ == "__main__":
    # Expanded training set
    def generate_combinations():
        combinations = []
        max_val = 20  # Extended to 20
        # Generate all pairs up to max_val
        for a in range(max_val + 1):
            for b in range(max_val + 1):
                combinations.append((a, b, a + b))
        # Add edge cases
        edge_cases = [
            (0, max_val, max_val),
            (max_val, max_val, max_val + max_val),
            (15, 15, 30),  # Numbers with many 1s in binary
        ]
        combinations.extend(edge_cases)
        # Randomly sample to keep size manageable
        if len(combinations) > 500:
            combinations = random.sample(combinations, 500)
        return combinations
    
    # Generate training set
    training_sets = generate_combinations()
    
    # Train model
    print("Training model...")
    logic_tables, final_logic_table, bits = train(
        training_sets,
        epochs=300,
        base_lr_min=0.05,
        base_lr_max=0.2,
        batch_size=32,
        perturbation_rate=0.01,
        regularization=0.001
    )
    
    # Print final logic tables
    for idx, table in enumerate(logic_tables):
        print(f"\nLogic Table for Binary2 Digit {idx+1}:")
        print("A\tB\tC\tOutput1_P0\tOutput1_P1\tOutput2_P0\tOutput2_P1")
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    print(f"{a}\t{b}\t{c}\t{1-table[a,b,c,0]:.2f}\t\t{table[a,b,c,0]:.2f}\t\t{1-table[a,b,c,1]:.2f}\t\t{table[a,b,c,1]:.2f}")
    
    print("\nFinal Logic Table:")
    print("A\tB\tC\tOutput1_P0\tOutput1_P1\tOutput2_P0\tOutput2_P1")
    for a in range(2):
        for b in range(2):
            for c in range(2):
                print(f"{a}\t{b}\t{c}\t{1-final_logic_table[a,b,c,0]:.2f}\t\t{final_logic_table[a,b,c,0]:.2f}\t\t{1-final_logic_table[a,b,c,1]:.2f}\t\t{final_logic_table[a,b,c,1]:.2f}")
    
    # Test predictions
    test_cases = [(2, 20), (5, 7), (15, 15)]
    for a, b in test_cases:
        print(f"\nPrediction for A={a}, B={b} (firm=True):")
        predicted_output = predict(a, b, logic_tables, final_logic_table, bits, firm=True)
        print(f"\nInput A: {a} (binary: {format(a, '0' + str(bits) + 'b')}), B: {b} (binary: {format(b, '0' + str(bits) + 'b')})")
        print(f"Predicted Output: {predicted_output} (binary: {format(predicted_output, '0' + str(bits+1) + 'b')})")
