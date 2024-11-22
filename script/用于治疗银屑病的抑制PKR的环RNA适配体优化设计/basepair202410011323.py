def extract_base_pairs(sequence, secondary_structure):
    """Extract paired bases from RNA sequence based on secondary structure."""
    stack = []
    pairs = []

    for i, char in enumerate(secondary_structure):
        if char == '(':
            stack.append(i)  # Push the index of the opening bracket
        elif char == ')':
            if stack:
                opening_index = stack.pop()  # Get the matching opening bracket index
                base1 = sequence[opening_index]
                base2 = sequence[i]
                pairs.append((opening_index + 1, base1, i + 1, base2))  # +1 for 1-based indexing

    return pairs

# Example usage
sequence = "AAAAUCCGUUGACCGGCAGCCAACUCUGCACAAAAUGUCCAUGAUGGGGCAUCGGGUCCGCAUUCUCCCAUGGUCUACCUUUCGCUUGAAUCUUAGUGUGACAACUCCGUACAAUGCAGACUUUGACGGGGAUGAGAUGAACUUGCACCUGCCACAGUCUCUGGAGACGCGAGCAGAGAUCCAGGAGCUGGCCAUGGUUCCUCGCAUGAUUGUCACCCCCCAGAGCAAUCGGCCUGUCAUGGGUAUUGUGCAGGACACACUCACAGCAGUGCGCAAAUUCACCAAGAGAGACGUCUUCCUGGAGCGGGUGGAACGGCACAUGUGUGAUGGGGACAUUGUUAUCUUCAAAGACGCUACGGACUU"
secondary_structure = "....(((((.....(((........((((((((....(((((((((((.(...((((..(((.((..((((((((......((((............))))...((((((.(((........)))))))))..(((.....)))((.((((.....(((((((..........))))))).)))).)).))))))))..........)).))).))))....((....))).)))))))))))..))))))))..(((((........((((.....((((((.....((........))......))))))...))))..))))).((((((........)))))).....))))))))..."

base_pairs = extract_base_pairs(sequence, secondary_structure)

# Output all paired bases and their indices
rt = open("base_pair.dat", "w")
print("Index1 Base1 Index2 Base2")
for idx1, base1, idx2, base2 in base_pairs:
    rt.write(f"{base1}{idx1},{base2}{idx2}\n")
    print(f"{idx1} {base1} {idx2} {base2}")
rt.close()
# Print the total number of base pairs found
print(f"\nTotal base pairs found: {len(base_pairs)}")
