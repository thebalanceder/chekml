from collections import defaultdict
import random
import math

# Step 1: Training function to handle multiple sets and create probabilistic logic table
def train(training_sets):
    # Initialize data structures
    all_digit_tables = []
    logic_table = {f"{a}{b}{c}": {'Output1': [], 'Output2': []} for a in '01' for b in '01' for c in '01'}
    carry_counts = defaultdict(lambda: {'0': 0, '1': 0})  # Track carry frequencies for [A,B,C]
    
    # Process each training set
    for decimal_a, decimal_b, output_decimal in training_sets:
        # Clamp negative outputs to 0
        output_decimal = max(0, int(output_decimal))
        # Calculate bits as max digits of inputs and output plus 1
        bits = max(
            math.ceil(math.log2(decimal_a + 1)) if decimal_a > 0 else 1,
            math.ceil(math.log2(decimal_b + 1)) if decimal_b > 0 else 1,
            math.ceil(math.log2(output_decimal + 1)) if output_decimal > 0 else 1
        ) + 1
        
        # Convert inputs and output to binary
        bin_a = format(int(decimal_a), '0' + str(bits) + 'b')
        bin_b = format(int(decimal_b), '0' + str(bits) + 'b')
        bin_out = format(output_decimal, '0' + str(bits) + 'b')
        
        # Create digit table with initial carry assignments
        digit_table = []
        carry = '0'  # Initialize C=0 for first digit
        for i in range(bits-1, -1, -1):  # LSB to MSB
            a_bit = bin_a[i]
            b_bit = bin_b[i]
            out_bit = bin_out[i]
            # Initial carry inference: try both 0 and 1, prefer based on output consistency
            if i > 0:
                # Check if output bit aligns with A, B, C sum
                sum_current = (1 if a_bit == '1' else 0) + (1 if b_bit == '1' else 0) + (1 if carry == '1' else 0)
                expected_out = sum_current % 2
                next_carry = '1' if out_bit != str(expected_out) else '0'
            else:
                next_carry = '0'  # No carry-out for last digit
            digit_table.append({
                'digit': bits-i,
                'A': a_bit,
                'B': b_bit,
                'C': carry,
                'Output1': next_carry,
                'Output2': out_bit,
                'Output1_confirmed': False,  # Mark as tentative
                'Output2_confirmed': True
            })
            carry = next_carry
        
        # Refine carries statistically
        for i in range(len(digit_table)-1):  # Exclude last digit
            entry = digit_table[i]
            abc = f"{entry['A']}{entry['B']}{entry['C']}"
            # Update carry counts based on initial assignment
            carry_counts[abc][entry['Output1']] += 1
            # Reassign Output1 based on majority vote
            if carry_counts[abc]['1'] > carry_counts[abc]['0']:
                entry['Output1'] = '1'
            else:
                entry['Output1'] = '0'
            entry['Output1_confirmed'] = True
            # Update next digit's C
            if i < len(digit_table)-1:
                digit_table[i+1]['C'] = entry['Output1']
        
        all_digit_tables.append(digit_table)
        
        # Update logic table with confirmed values
        for entry in digit_table:
            abc = f"{entry['A']}{entry['B']}{entry['C']}"
            if entry['Output2_confirmed']:
                logic_table[abc]['Output2'].append(entry['Output2'])
            if entry['Output1_confirmed']:
                logic_table[abc]['Output1'].append(entry['Output1'])
    
    # Compute probabilities for qubits in logic table
    prob_logic_table = {}
    for abc, outputs in logic_table.items():
        p1_out1 = sum(1 for v in outputs['Output1'] if v == '1') / len(outputs['Output1']) if outputs['Output1'] else 0.01
        p1_out2 = sum(1 for v in outputs['Output2'] if v == '1') / len(outputs['Output2']) if outputs['Output2'] else 0.01
        prob_logic_table[abc] = {
            'Output1_prob': {'P0': 1.0 - p1_out1, 'P1': p1_out1},
            'Output2_prob': {'P0': 1.0 - p1_out2, 'P1': p1_out2}
        }
    
    return all_digit_tables, prob_logic_table

# Step 2: Helper function to compute qubit probabilities
def compute_prob(values):
    if not values:  # Unrecorded [A,B,C]
        return {'P0': 1.0, 'P1': 0.01}  # Small P1 for robustness
    count_1 = sum(1 for v in values if v == '1')
    total = len(values)
    p1 = count_1 / total if total > 0 else 0.01
    return {'P0': 1.0 - p1, 'P1': p1}

# Step 3: Prediction function with additional digit and firm option
def predict(decimal_a, decimal_b, prob_logic_table, firm=False):
    # Calculate bits as max digits of inputs plus 1
    bits = max(
        math.ceil(math.log2(decimal_a + 1)) if decimal_a > 0 else 1,
        math.ceil(math.log2(decimal_b + 1)) if decimal_b > 0 else 1
    ) + 1
    
    # Convert decimals to binary (pad to specified bits)
    bin_a = format(int(decimal_a), '0' + str(bits) + 'b')
    bin_b = format(int(decimal_b), '0' + str(bits) + 'b')
    
    result = []
    carry = '0'  # Initialize C=0
    
    # Process standard digits (LSB to MSB)
    for i in range(bits-1, -1, -1):
        a_bit = bin_a[i]
        b_bit = bin_b[i]
        abc = f"{a_bit}{b_bit}{carry}"
        # Get probabilities from logic table
        probs = prob_logic_table[abc]
        if firm:
            # Choose most probable state
            out1 = '1' if probs['Output1_prob']['P1'] > probs['Output1_prob']['P0'] else '0'
            out2 = '1' if probs['Output2_prob']['P1'] > probs['Output2_prob']['P0'] else '0'
        else:
            # Sample probabilistically
            out1 = '1' if random.random() < probs['Output1_prob']['P1'] else '0'
            out2 = '1' if random.random() < probs['Output2_prob']['P1'] else '0'
        result.append(out2)
        carry = out1  # Update carry for next digit
        print(f"Digit {bits-i}: A={a_bit}, B={b_bit}, C={carry}, Output1={out1} (P1={probs['Output1_prob']['P1']}), Output2={out2} (P1={probs['Output2_prob']['P1']})")
    
    # Process additional digit (A=0, B=0, C=previous Output1)
    a_bit = '0'
    b_bit = '0'
    abc = f"{a_bit}{b_bit}{carry}"
    probs = prob_logic_table[abc]
    if firm:
        out1 = '1' if probs['Output1_prob']['P1'] > probs['Output1_prob']['P0'] else '0'
        out2 = '1' if probs['Output2_prob']['P1'] > probs['Output2_prob']['P0'] else '0'
    else:
        out1 = '1' if random.random() < probs['Output1_prob']['P1'] else '0'
        out2 = '1' if random.random() < probs['Output2_prob']['P1'] else '0'
    result.append(out2)
    print(f"Digit {bits+1} (Extra): A={a_bit}, B={b_bit}, C={out1}, Output1={out1} (P1={probs['Output1_prob']['P1']}), Output2={out2} (P1={probs['Output2_prob']['P1']})")
    
    # Reverse result to get correct binary order
    result = result[::-1]
    print(f"Result bits (MSB to LSB): {result}")
    # Convert binary result to decimal
    binary_str = ''.join(result)
    print(f"Binary string: {binary_str}")
    decimal_output = int(binary_str, 2)
    
    return decimal_output

# Step 4: Example usage
if __name__ == "__main__":
    # Define multiple training sets
    def generate_combinations():
        combinations = []
        for a in range(101):  # First number from 0 to 100
            for b in range(101):  # Second number from 0 to 100
                combinations.append((a, b, a + b - 2))
        # Add specific test case
        combinations.append((2, 20, 20))
        return combinations

    # Example usage:
    result = generate_combinations()
    training_sets = result
    
    # Train model
    digit_tables, prob_logic_table = train(training_sets)
    
    # Print probabilistic logic table
    print("\nProbabilistic Logic Table:")
    print("A\tB\tC\tOutput1_P0\tOutput1_P1\tOutput2_P0\tOutput2_P1")
    for abc, probs in prob_logic_table.items():
        print(f"{abc[0]}\t{abc[1]}\t{abc[2]}\t{probs['Output1_prob']['P0']:.2f}\t\t{probs['Output1_prob']['P1']:.2f}\t\t{probs['Output2_prob']['P0']:.2f}\t\t{probs['Output2_prob']['P1']:.2f}")
    
    # Predict for A=2, B=20 with firm=True
    print(f"\nPrediction for A=2, B=20 (firm=True):")
    predicted_output = predict(2, 20, prob_logic_table, firm=True)
    print(f"\nInput A: 2 (binary: {format(2, '06b')}), B: 20 (binary: {format(20, '06b')})")
    print(f"Predicted Output: {predicted_output} (binary: {format(predicted_output, '07b')})")
