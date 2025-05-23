import pandas as pd 
from mendeleev import element
from pprint import pprint
import json 

df = pd.read_csv('./Data/Informative_table_21.05.final_filter.csv', delimiter=';')
configs = df[['Pure metals', 'Electronegativity', "Atomic radius (pm)", "Melting temperature (°C)"]]

metals_dict = {}

for _, row in configs.iterrows():
    metal_symbol = row['Pure metals']
    metals_dict[metal_symbol] = {
                    "atomic_number": element(metal_symbol).atomic_number,
                    "electronegativity": float(row['Electronegativity'].replace(',', '.')),
                    "atomic_radius": float(row['Atomic radius (pm)']),
                    "melting_point": float(row['Melting temperature (°C)'].replace(',', '.')),
                    # Add ionization energy when available
                    "ionization_energy": min(element(metal_symbol).ionenergies.values()) #minimum enegry value 
                }

    # print("Element info \n")
    # pprint(element_info)

    # break
print("\nHashtabel is \n")
pprint(metals_dict)
# Save the dictionary to a JSON file
with open('./configs/elements_vocab.json', 'w') as f:
    json.dump(metals_dict, f, indent=4)