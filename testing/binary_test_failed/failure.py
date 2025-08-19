import random
import math
import numpy as np

# Initialize a logic table as a NumPy array
def initialize_logic_table():
    # Shape: (2,2,2,2) for A,B,C,(Output1_P1, Output2_P1)
    table = np.full((2, 2, 2, 2), 0.5, dtype=np.float32)
    return table

# Training function to learn binary interactions
def train(training_sets, epochs=200, lr=0.1, batch_size=32):
    # Determine bits needed
    max_decimal = max(max(abs(int(a)), abs(int(b)), abs(int(out))) for a, b, out in training_sets)
    bits = math.ceil(math.log2(max_decimal + 1)) + 1
    
    # Initialize logic tables
    logic_tables = [initialize_logic_table() for _ in range(bits)]
    final_logic_table = initialize_logic_table()
    
    # Validation set
    validation_sets = training_sets[::5]
    training_sets = [t for t in training_sets if t not in validation_sets]
    
    best_val_error = float('inf')
    best_logic_tables = logic_tables
    best_final_table = final_logic_table
    
    for epoch in range(epochs):
        total_train_hamming = 0
        total_train_decimal = 0
        random.shuffle(training_sets)
        
        for batch_start in range(0, len(training_sets), batch_size):
            batch = training_sets[batch_start:batch_start + batch_size]
            
            for decimal_a, decimal_b, expected_output in batch:
                bin_a = format(int(decimal_a), '0' + str(bits) + 'b')
                bin_b = format(int(decimal_b), '0' + str(bits) + 'b')
                bin_expected = format(max(0, int(expected_output)), '0' + str(bits+1) + 'b')
                
                # Expected carries (simplified: assume carry patterns to be learned)
                expected_carries = [[] for _ in range(bits)]
                for table_idx in range(bits):
                    for pos in range(bits):
                        expected_carries[table_idx].append('0')  # Placeholder, learned implicitly
                
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
                        weight = 2 ** (pos / bits)
                        grad = delta * weight
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 1] += lr * grad
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 1] = max(0.01, min(0.99, logic_tables[table_idx][a_idx, b_idx, c_idx, 1]))
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 0] = max(0.01, min(0.99, 1 - logic_tables[table_idx][a_idx, b_idx, c_idx, 1]))
                        
                        # Output1 (carry, simplified learning)
                        delta = 1.0 if random.random() < 0.5 else -1.0  # Random carry learning until pattern is clear
                        grad = delta * weight
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 0] += lr * grad * 0.5  # Lower weight for carries
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 0] = max(0.01, min(0.99, logic_tables[table_idx][a_idx, b_idx, c_idx, 0]))
                        logic_tables[table_idx][a_idx, b_idx, c_idx, 0] = max(0.01, min(0.99, logic_tables[table_idx][a_idx, b_idx, c_idx, 0]))
                
                for a_idx, b_idx, c_idx, out1, out2, pos in final_used_entries:
                    correct_out2 = bin_expected[pos] if pos < len(bin_expected) else '0'
                    delta = 1.0 if out2 == correct_out2 else -1.0
                    weight = 2 ** (pos / bits)
                    grad = delta * weight
                    final_logic_table[a_idx, b_idx, c_idx, 1] += lr * grad
                    final_logic_table[a_idx, b_idx, c_idx, 1] = max(0.01, min(0.99, final_logic_table[a_idx, b_idx, c_idx, 1]))
                    final_logic_table[a_idx, b_idx, c_idx, 0] = max(0.01, min(0.99, 1 - final_logic_table[a_idx, b_idx, c_idx, 1]))
                
        # Validation
        total_val_hamming = 0
        for decimal_a, decimal_b, expected_output in validation_sets:
            bin_expected = format(max(0, int(expected_output)), '0' + str(bits+1) + 'b')
            result = predict(decimal_a, decimal_b, logic_tables, final_logic_table, bits, firm=True, silent=True)
            result_bin = format(result, '0' + str(bits+1) + 'b')
            hamming_error = sum(a != b for a, b in zip(result_bin, bin_expected))
            total_val_hamming += hamming_error
        
        avg_train_hamming = total_train_hamming / len(training_sets)
        avg_train_decimal = total_train_decimal / len(training_sets)
        avg_val_hamming = total_val_hamming / len(validation_sets)
        
        if avg_val_hamming < best_val_error:
            best_val_error = avg_val_hamming
            best_logic_tables = [table.copy() for table in logic_tables]
            best_final_table = final_logic_table.copy()
        
        print(f"Epoch {epoch+1}, Train Hamming: {avg_train_hamming:.4f}, Train Decimal: {avg_train_decimal:.4f}, Val Hamming: {avg_val_hamming:.4f}")
    
    return best_logic_tables, best_final_table, bits

# Prediction function
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
    # Placeholder dataset (assuming addition; replace with actual data)
    def generate_combinations():
        combinations = []
        for a in range(101):  # 0 to 10
            for b in range(101):
                # Example: output is a + b (replace with your pattern)
                combinations.append((a, b, a + b))
        return combinations
    
    # Generate training set
    training_sets = generate_combinations()
    
    # Train model
    print("Training model...")
    logic_tables, final_logic_table, bits = train(training_sets, epochs=200, lr=0.1, batch_size=32)
    
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
    test_cases = [(2, 3), (5, 7), (10, 10)]
    for a, b in test_cases:
        print(f"\nPrediction for A={a}, B={b} (firm=True):")
        predicted_output = predict(a, b, logic_tables, final_logic_table, bits, firm=True)
        print(f"\nInput A: {a} (binary: {format(a, '0' + str(bits) + 'b')}), B: {b} (binary: {format(b, '0' + str(bits) + 'b')})")
        print(f"Predicted Output: {predicted_output} (binary: {format(predicted_output, '0' + str(bits+1) + 'b')})")
