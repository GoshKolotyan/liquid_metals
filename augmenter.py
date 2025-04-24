from helper import parse_compostion
from itertools import permutations
import pandas as pd

# Load the data
df = pd.read_csv('./Data/cleaned_data.csv')

# Prepare a list to store generated compositions
gens = []

for index, row in df.iterrows():
    composition = row['System_Name']
    try:
        target = float(row['Tm (Liquidus)'])  # Ensure target is a float
    except ValueError:
        print(f"Invalid target temperature at index {index}, skipping.")
        continue

    print("Composition:", type(composition), "Target:", type(target))
    print("Composition:", composition, "Target:", target)

    # Parse the composition
    parsed = parse_compostion(composition)  # Expected output: [('Si', 0.1), ('Al', 0.9)]
    print("Parsed composition:", parsed)

    # Filter out invalid entries
    parsed = [item for item in parsed if isinstance(item, tuple) and item[0] and item[1] != 0.0]

    # Generate all permutations
    for per in permutations(parsed):
        res = ''
        for comp, perc in per:
            res += f"{comp}" if perc == 1 else f"{comp}{perc}"
        gens.append({'Generated_Composition': res, 'Target_Temperature': target})

# Convert to DataFrame
generated_df = pd.DataFrame(gens)

# Save to CSV
generated_df.to_csv('./Data/generated_compositions.csv', index=False)
print("Saved generated compositions to './Data/generated_compositions.csv'")
