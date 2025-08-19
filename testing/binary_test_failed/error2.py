from collections import defaultdict
import random

# Step 1: Training function to handle multiple sets and create probabilistic logic table
def train(training_sets, bits=4):
    # Initialize data structures
    all_digit_tables = []
    logic_table = {f"{a}{b}{c}": {'Output1': [], 'Output2': []} for a in '01' for b in '01' for c in '01'}
    
    # Process each training set
    for decimal_a, decimal_b, output_decimal in training_sets:
        # Convert inputs to binary
        bin_a = format(int(decimal_a), '0' + str(bits) + 'b')
        bin_b = format(int(decimal_b), '0' + str(bits) + 'b')
        # Clamp negative outputs to 0
        output_decimal = max(0, int(output_decimal))
        bin_out = format(output_decimal, '0' + str(bits) + 'b')
        
        # Create digit table
        digit_table = []
        carry = '0'  # Initialize C=0 for first digit
        for i in range(bits-1, -1, -1):  # LSB to MSB
            a_bit = bin_a[i]
            b_bit = bin_b[i]
            out_bit = bin_out[i]
            # Output1 is C of next digit, use placeholder initially
            next_carry = '0' if i == 0 else 'X' + str(bits-i)
            digit_table.append({
                'digit': bits-i,
                'A': a_bit,
                'B': b_bit,
                'C': carry,
                'Output1': next_carry,
                'Output2': out_bit,
                'Output1_confirmed': False  # Track if Output1 is confirmed
            })
            carry = next_carry  # Next C is current Output1
        
        # Apply rule: if A=0, B=0, Output2=1, set C=1 and previous Output1=1
        for i, entry in enumerate(digit_table):
            if entry['A'] == '0' and entry['B'] == '0' and entry['Output2'] == '1':
                entry['C'] = '1'  # Set C=1 for this digit
                entry['C_confirmed'] = True  # Mark C as confirmed
                # Update previous digit's Output1 to 1 (if not the first digit)
                if i > 0:
                    digit_table[i-1]['Output1'] = '1'
                    digit_table[i-1]['Output1_confirmed'] = True
        
        # Check for redundant [AB] with different Output2
        ab_out2 = defaultdict(list)
        for entry in digit_table:
            ab = entry['A'] + entry['B']
            ab_out2[ab].append((entry['digit'], entry['Output2']))
        
        # Assign C values for redundant [AB] with different Output2
        for ab, digits in ab_out2.items():
            if len(digits) > 1:
                out2_values = [d[1] for d in digits]
                if len(set(out2_values)) > 1:  # Different Output2
                    # Assign different C values (0 and 1)
                    for i, (digit, _) in enumerate(digits):
                        c_val = '0' if i == 0 else '1'
                        # Update C for this digit
                        for entry in digit_table:
                            if entry['digit'] == digit:
                                entry['C'] = c_val
                                entry['C_confirmed'] = True
                                # Update previous digit's Output1
                                if digit > 1:
                                    for prev_entry in digit_table:
                                        if prev_entry['digit'] == digit - 1:
                                            prev_entry['Output1'] = c_val
                                            prev_entry['Output1_confirmed'] = True
        
        # Resolve unconfirmed C and Output1 values
        for i, entry in enumerate(digit_table):
            if entry['C'].startswith('X'):
                # Default unconfirmed C to 0
                entry['C'] = '0'
                entry['C_confirmed'] = False  # Mark as unconfirmed
                # Update previous digit's Output1 to 0 (unconfirmed)
                if i > 0:
                    digit_table[i-1]['Output1'] = '0'
                    digit_table[i-1]['Output1_confirmed'] = False
            # Mark last digit's Output1 as confirmed (set to 0, no next digit)
            if entry['digit'] == bits:
                entry['Output1_confirmed'] = True
        
        all_digit_tables.append(digit_table)
        
        # Update logic table with confirmed values only
        for entry in digit_table:
            abc = f"{entry['A']}{entry['B']}{entry['C']}"
            # Output2 is always confirmed (from training output)
            logic_table[abc]['Output2'].append(entry['Output2'])
            # Output1 only if confirmed
            if entry['Output1_confirmed']:
                logic_table[abc]['Output1'].append(entry['Output1'])
    
    # Compute probabilities for qubits in logic table
    prob_logic_table = {}
    for abc, outputs in logic_table.items():
        prob_logic_table[abc] = {
            'Output1_prob': compute_prob(outputs['Output1']),
            'Output2_prob': compute_prob(outputs['Output2'])
        }
    
    return all_digit_tables, prob_logic_table

# Step 2: Helper function to compute qubit probabilities
def compute_prob(values):
    if not values:  # Unrecorded [A,B,C]
        return {'P0': 1.0, 'P1': 0.0}  # Default to 0
    count_1 = sum(1 for v in values if v == '1')
    total = len(values)
    p1 = count_1 / total if total > 0 else 0.0
    return {'P0': 1.0 - p1, 'P1': p1}

# Step 3: Prediction function with additional digit and firm option
def predict(decimal_a, decimal_b, prob_logic_table, bits=4, firm=False):
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
        for a in range(11):          # First number from 0 to 10
            for b in range(11):      # Second number from 0 to 10
                combinations.append((a, b, a + b ))
        return combinations

    # Example usage:
    result = generate_combinations()

    training_sets = result
    
    # Train model
    digit_tables, prob_logic_table = train(training_sets)
    
    # Print digit tables
    for idx, digit_table in enumerate(digit_tables, 1):
        print(f"\nDigit Table for Training Set {idx}:")
        print("Digit\tA\tB\tC\tOutput1\tOutput2\tOutput1_Confirmed")
        for entry in digit_table:
            print(f"{entry['digit']}\t{entry['A']}\t{entry['B']}\t{entry['C']}\t{entry['Output1']}\t{entry['Output2']}\t{entry['Output1_confirmed']}")
    
    # Print probabilistic logic table
    print("\nProbabilistic Logic Table:")
    print("A\tB\tC\tOutput1_P0\tOutput1_P1\tOutput2_P0\tOutput2_P1")
    for abc, probs in prob_logic_table.items():
        print(f"{abc[0]}\t{abc[1]}\t{abc[2]}\t{probs['Output1_prob']['P0']:.2f}\t\t{probs['Output1_prob']['P1']:.2f}\t\t{probs['Output2_prob']['P0']:.2f}\t\t{probs['Output2_prob']['P1']:.2f}")
   
    a=2
    b=20
    # Predict for AB with firm=True
    print(f"\nPrediction for A={a}, B={b} (firm=True):")
    predicted_output = predict(a, b, prob_logic_table, firm=True)
    print(f"\nInput A: {a} (binary: {format(a, '04b')}), B: {b} (binary: {format(b, '04b')})")
    print(f"Predicted Output: {predicted_output} (binary: {format(predicted_output, '05b')})")
