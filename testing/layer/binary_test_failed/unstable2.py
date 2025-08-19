from collections import defaultdict
import random
import math
import numpy as np

# Initialize a logic table with 50%/50% probabilities for Output1 and Output2
def initialize_logic_table():
    logic_table = {f"{a}{b}{c}": {'Output1_prob': {'P0': 0.5, 'P1': 0.5}, 'Output2_prob': {'P0': 0.5, 'P1': 0.5}}
                   for a in '01' for b in '01' for c in '01'}
    return logic_table

# Training function with enhanced learning
def train(training_sets, epochs=200, base_learning_rate=0.1, perturbation_rate=0.01):
    # Determine bits needed
    max_decimal = max(max(abs(int(a)), abs(int(b)), abs(int(out))) for a, b, out in training_sets)
    bits = math.ceil(math.log2(max_decimal + 1)) + 1
    
    # Initialize logic tables
    logic_tables = [initialize_logic_table() for _ in range(bits)]
    final_logic_table = initialize_logic_table()
    
    # For validation
    validation_sets = training_sets[::5]  # Every 5th example for validation
    training_sets = [t for t in training_sets if t not in validation_sets]
    
    best_val_error = float('inf')
    best_logic_tables = logic_tables
    best_final_table = final_logic_table
    
    for epoch in range(epochs):
        total_train_error = 0
        total_train_decimal_error = 0
        
        # Shuffle training sets
        random.shuffle(training_sets)
        
        # Adaptive learning rate
        lr = base_learning_rate * (1.0 / (1.0 + 0.01 * epoch))
        
        for decimal_a, decimal_b, expected_output in training_sets:
            # Convert to binary
            bin_a = format(int(decimal_a), '0' + str(bits) + 'b')
            bin_b = format(int(decimal_b), '0' + str(bits) + 'b')
            bin_expected = format(max(0, int(expected_output)), '0' + str(bits+1) + 'b')
            
            # Simulate prediction
            hidden_binaries = []
            used_entries = [[] for _ in range(bits)]  # Track (abc, out1, out2, pos)
            final_used_entries = []
            
            # Expected carries for training Output1
            expected_carries = []
            carry = 0
            for i in range(bits-1, -1, -1):
                a_bit = int(bin_a[i])
                b_bit = int(bin_b[i])
                sum_bit = a_bit + b_bit + carry
                expected_carries.append(str(sum_bit // 2))  # Carry out
                carry = sum_bit // 2
            expected_carries = expected_carries[::-1]  # MSB to LSB
            
            # Process binary2 digits
            for digit_b_idx in range(bits-1, -1, -1):
                b_bit = bin_b[digit_b_idx]
                current_table = logic_tables[bits-1-digit_b_idx]
                hidden_binary = []
                carry = '0'
                for digit_a_idx in range(bits-1, -1, -1):
                    a_bit = bin_a[digit_a_idx]
                    abc = f"{a_bit}{b_bit}{carry}"
                    probs = current_table[abc]
                    out1 = '1' if random.random() < probs['Output1_prob']['P1'] else '0'
                    out2 = '1' if random.random() < probs['Output2_prob']['P1'] else '0'
                    hidden_binary.append(out2)
                    carry = out1
                    pos = bits - 1 - digit_a_idx
                    used_entries[bits-1-digit_b_idx].append((abc, out1, out2, pos))
                hidden_binary = hidden_binary[::-1]
                if (bits-1-digit_b_idx) > 0:
                    shift = bits-1-digit_b_idx
                    hidden_binary = ['0'] * shift + hidden_binary[:-shift]
                hidden_binaries.append(hidden_binary)
            
            # Final layer
            result = []
            carry = '0'
            for pos in range(bits):
                bits_at_pos = [hb[pos] for hb in hidden_binaries]
                a_bit = bits_at_pos[0] if len(bits_at_pos) > 0 else '0'
                b_bit = bits_at_pos[1] if len(bits_at_pos) > 1 else '0'
                abc = f"{a_bit}{b_bit}{carry}"
                probs = final_logic_table[abc]
                out1 = '1' if random.random() < probs['Output1_prob']['P1'] else '0'
                out2 = '1' if random.random() < probs['Output2_prob']['P1'] else '0'
                result.append(out2)
                carry = out1
                final_used_entries.append((abc, out1, out2, pos))
            # Extra digit
            a_bit = '0'
            b_bit = '0'
            abc = f"{a_bit}{b_bit}{carry}"
            probs = final_logic_table[abc]
            out1 = '1' if random.random() < probs['Output1_prob']['P1'] else '0'
            out2 = '1' if random.random() < probs['Output2_prob']['P1'] else '0'
            result.append(out2)
            final_used_entries.append((abc, out1, out2, bits))
            result = result[::-1]
            
            # Compute errors
            hamming_error = sum(a != b for a, b in zip(''.join(result), bin_expected))
            predicted_decimal = int(''.join(result), 2)
            decimal_error = abs(predicted_decimal - int(expected_output))
            total_train_error += hamming_error
            total_train_decimal_error += decimal_error
            
            # Update probabilities using matrix operations
            for table_idx, entries in enumerate(used_entries):
                for abc, out1, out2, pos in entries:
                    # Update Output2
                    correct_out2 = bin_expected[pos] if pos < len(bin_expected) else '0'
                    delta = 1.0 if out2 == correct_out2 else -1.0
                    weight = 1.0 / (1.0 + decimal_error)  # Weight updates by error
                    p1 = logic_tables[table_idx][abc]['Output2_prob']['P1']
                    logic_tables[table_idx][abc]['Output2_prob']['P1'] += lr * delta * weight * (1 - p1 if delta > 0 else p1)
                    logic_tables[table_idx][abc]['Output2_prob']['P1'] = max(0.01, min(0.99, logic_tables[table_idx][abc]['Output2_prob']['P1']))
                    logic_tables[table_idx][abc]['Output2_prob']['P0'] = 1 - logic_tables[table_idx][abc]['Output2_prob']['P1']
                    
                    # Update Output1
                    correct_out1 = expected_carries[pos] if pos < len(expected_carries) else '0'
                    delta = 1.0 if out1 == correct_out1 else -1.0
                    p1 = logic_tables[table_idx][abc]['Output1_prob']['P1']
                    logic_tables[table_idx][abc]['Output1_prob']['P1'] += lr * delta * weight * (1 - p1 if delta > 0 else p1)
                    logic_tables[table_idx][abc]['Output1_prob']['P1'] = max(0.01, min(0.99, logic_tables[table_idx][abc]['Output1_prob']['P1']))
                    logic_tables[table_idx][abc]['Output1_prob']['P0'] = 1 - logic_tables[table_idx][abc]['Output1_prob']['P0']
            
            for abc, out1, out2, pos in final_used_entries:
                correct_out2 = bin_expected[pos] if pos < len(bin_expected) else '0'
                delta = 1.0 if out2 == correct_out2 else -1.0
                weight = 1.0 / (1.0 + decimal_error)
                p1 = final_logic_table[abc]['Output2_prob']['P1']
                final_logic_table[abc]['Output2_prob']['P1'] += lr * delta * weight * (1 - p1 if delta > 0 else p1)
                final_logic_table[abc]['Output2_prob']['P1'] = max(0.01, min(0.99, final_logic_table[abc]['Output2_prob']['P1']))
                final_logic_table[abc]['Output2_prob']['P0'] = 1 - final_logic_table[abc]['Output2_prob']['P0']
                
                # Update Output1 (simplified, as final layer carry is less critical)
                p1 = final_logic_table[abc]['Output1_prob']['P1']
                final_logic_table[abc]['Output1_prob']['P1'] = max(0.01, min(0.99, p1))
                final_logic_table[abc]['Output1_prob']['P0'] = 1 - final_logic_table[abc]['Output1_prob']['P1']
            
            # Perturbation for exploration
            if random.random() < perturbation_rate:
                for table in logic_tables:
                    for abc in table:
                        table[abc]['Output1_prob']['P1'] += random.uniform(-0.05, 0.05)
                        table[abc]['Output1_prob']['P1'] = max(0.01, min(0.99, table[abc]['Output1_prob']['P1']))
                        table[abc]['Output1_prob']['P0'] = 1 - table[abc]['Output1_prob']['P1']
                        table[abc]['Output2_prob']['P1'] += random.uniform(-0.05, 0.05)
                        table[abc]['Output2_prob']['P1'] = max(0.01, min(0.99, table[abc]['Output2_prob']['P1']))
                        table[abc]['Output2_prob']['P0'] = 1 - table[abc]['Output2_prob']['P0']
                for abc in final_logic_table:
                    final_logic_table[abc]['Output1_prob']['P1'] += random.uniform(-0.05, 0.05)
                    final_logic_table[abc]['Output1_prob']['P1'] = max(0.01, min(0.99, final_logic_table[abc]['Output1_prob']['P1']))
                    final_logic_table[abc]['Output1_prob']['P0'] = 1 - final_logic_table[abc]['Output1_prob']['P1']
                    final_logic_table[abc]['Output2_prob']['P1'] += random.uniform(-0.05, 0.05)
                    final_logic_table[abc]['Output2_prob']['P1'] = max(0.01, min(0.99, final_logic_table[abc]['Output2_prob']['P1']))
                    final_logic_table[abc]['Output2_prob']['P0'] = 1 - final_logic_table[abc]['Output2_prob']['P0']
        
        # Validation
        total_val_error = 0
        for decimal_a, decimal_b, expected_output in validation_sets:
            bin_a = format(int(decimal_a), '0' + str(bits) + 'b')
            bin_b = format(int(decimal_b), '0' + str(bits) + 'b')
            bin_expected = format(max(0, int(expected_output)), '0' + str(bits+1) + 'b')
            result = predict(decimal_a, decimal_b, logic_tables, final_logic_table, bits, firm=True, silent=True)
            result_bin = format(result, '0' + str(bits+1) + 'b')
            hamming_error = sum(a != b for a, b in zip(result_bin, bin_expected))
            total_val_error += hamming_error
        
        avg_train_error = total_train_error / len(training_sets)
        avg_train_decimal_error = total_train_decimal_error / len(training_sets)
        avg_val_error = total_val_error / len(validation_sets)
        
        # Save best model
        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            best_logic_tables = [dict(table) for table in logic_tables]
            best_final_table = dict(final_logic_table)
        
        print(f"Epoch {epoch+1}, Train Hamming Error: {avg_train_error:.4f}, Train Decimal Error: {avg_train_decimal_error:.4f}, Val Hamming Error: {avg_val_error:.4f}")
    
    return best_logic_tables, best_final_table, bits

# Prediction function (unchanged)
def predict(decimal_a, decimal_b, logic_tables, final_logic_table, bits, firm=False, silent=False):
    bin_a = format(int(decimal_a), '0' + str(bits) + 'b')
    bin_b = format(int(decimal_b), '0' + str(bits) + 'b')
    
    hidden_binaries = []
    
    for digit_b_idx in range(bits-1, -1, -1):
        b_bit = bin_b[digit_b_idx]
        current_logic_table = logic_tables[bits-1-digit_b_idx]
        hidden_binary = []
        carry = '0'
        
        for digit_a_idx in range(bits-1, -1, -1):
            a_bit = bin_a[digit_a_idx]
            abc = f"{a_bit}{b_bit}{carry}"
            probs = current_logic_table[abc]
            
            if firm:
                out1 = '1' if probs['Output1_prob']['P1'] > probs['Output1_prob']['P0'] else '0'
                out2 = '1' if probs['Output2_prob']['P1'] > probs['Output2_prob']['P0'] else '0'
            else:
                out1 = '1' if random.random() < probs['Output1_prob']['P1'] else '0'
                out2 = '1' if random.random() < probs['Output2_prob']['P1'] else '0'
            
            hidden_binary.append(out2)
            carry = out1
            if not silent:
                print(f"Binary2 Digit {bits-digit_b_idx}: A={a_bit}, B={b_bit}, C={carry}, Output1={out1} (P1={probs['Output1_prob']['P1']:.2f}), Output2={out2} (P1={probs['Output2_prob']['P1']:.2f})")
        
        hidden_binary = hidden_binary[::-1]
        shift_amount = bits - 1 - digit_b_idx
        if shift_amount > 0:
            hidden_binary = ['0'] * shift_amount + hidden_binary[:-shift_amount]
        hidden_binaries.append(hidden_binary)
        if not silent:
            print(f"Hidden Binary {bits-digit_b_idx}: {hidden_binary}")
    
    result = []
    carry = '0'
    for pos in range(bits):
        bits_at_pos = [hb[pos] for hb in hidden_binaries]
        a_bit = bits_at_pos[0] if len(bits_at_pos) > 0 else '0'
        b_bit = bits_at_pos[1] if len(bits_at_pos) > 1 else '0'
        abc = f"{a_bit}{b_bit}{carry}"
        probs = final_logic_table[abc]
        
        if firm:
            out1 = '1' if probs['Output1_prob']['P1'] > probs['Output1_prob']['P0'] else '0'
            out2 = '1' if probs['Output2_prob']['P1'] > probs['Output2_prob']['P0'] else '0'
        else:
            out1 = '1' if random.random() < probs['Output1_prob']['P1'] else '0'
            out2 = '1' if random.random() < probs['Output2_prob']['P1'] else '0'
        
        result.append(out2)
        carry = out1
        if not silent:
            print(f"Final Layer Pos {pos+1}: A={a_bit}, B={b_bit}, C={carry}, Output1={out1} (P1={probs['Output1_prob']['P1']:.2f}), Output2={out2} (P1={probs['Output2_prob']['P1']:.2f})")
    
    a_bit = '0'
    b_bit = '0'
    abc = f"{a_bit}{b_bit}{carry}"
    probs = final_logic_table[abc]
    if firm:
        out1 = '1' if probs['Output1_prob']['P1'] > probs['Output1_prob']['P0'] else '0'
        out2 = '1' if probs['Output2_prob']['P1'] > probs['Output2_prob']['P0'] else '0'
    else:
        out1 = '1' if random.random() < probs['Output1_prob']['P1'] else '0'
        out2 = '1' if random.random() < probs['Output2_prob']['P1'] else '0'
    result.append(out2)
    if not silent:
        print(f"Final Layer Extra Digit: A={a_bit}, B={b_bit}, C={out1}, Output1={out1} (P1={probs['Output1_prob']['P1']:.2f}), Output2={out2} (P1={probs['Output2_prob']['P1']:.2f})")
    
    result = result[::-1]
    if not silent:
        print(f"Result bits (MSB to LSB): {result}")
        binary_str = ''.join(result)
        print(f"Binary string: {binary_str}")
    decimal_output = int(''.join(result), 2)
    
    return decimal_output

# Example usage
if __name__ == "__main__":
    # Define multiple training sets
    def generate_combinations():
        combinations = []
        for a in range(11):  # First number from 0 to 10
            for b in range(11):  # Second number from 0 to 10
                combinations.append((a, b, a + b))
        return combinations
    
    # Generate training set
    training_sets = generate_combinations()
    
    # Train model
    print("Training model...")
    logic_tables, final_logic_table, bits = train(training_sets, epochs=200, base_learning_rate=0.1, perturbation_rate=0.01)
    
    # Print final logic tables
    for idx, table in enumerate(logic_tables):
        print(f"\nLogic Table for Binary2 Digit {idx+1}:")
        print("A\tB\tC\tOutput1_P0\tOutput1_P1\tOutput2_P0\tOutput2_P1")
        for abc, probs in table.items():
            print(f"{abc[0]}\t{abc[1]}\t{abc[2]}\t{probs['Output1_prob']['P0']:.2f}\t\t{probs['Output1_prob']['P1']:.2f}\t\t{probs['Output2_prob']['P0']:.2f}\t\t{probs['Output2_prob']['P1']:.2f}")
    
    print("\nFinal Logic Table:")
    print("A\tB\tC\tOutput1_P0\tOutput1_P1\tOutput2_P0\tOutput2_P1")
    for abc, probs in final_logic_table.items():
        print(f"{abc[0]}\t{abc[1]}\t{abc[2]}\t{probs['Output1_prob']['P0']:.2f}\t\t{probs['Output1_prob']['P1']:.2f}\t\t{probs['Output2_prob']['P0']:.2f}\t\t{probs['Output2_prob']['P1']:.2f}")
    
    # Test prediction
    a = 2
    b = 20
    print(f"\nPrediction for A={a}, B={b} (firm=True):")
    predicted_output = predict(a, b, logic_tables, final_logic_table, bits, firm=True)
    print(f"\nInput A: {a} (binary: {format(a, '0' + str(bits) + 'b')}), B: {b} (binary: {format(b, '0' + str(bits) + 'b')})")
    print(f"Predicted Output: {predicted_output} (binary: {format(predicted_output, '0' + str(bits+1) + 'b')})")
    
    # Additional test within training range
    a = 5
    b = 7
    print(f"\nPrediction for A={a}, B={b} (firm=True):")
    predicted_output = predict(a, b, logic_tables, final_logic_table, bits, firm=True)
    print(f"\nInput A: {a} (binary: {format(a, '0' + str(bits) + 'b')}), B: {b} (binary: {format(b, '0' + str(bits) + 'b')})")
    print(f"Predicted Output: {predicted_output} (binary: {format(predicted_output, '0' + str(bits+1) + 'b')})")
