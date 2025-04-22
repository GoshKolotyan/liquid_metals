import re 



def parse_compostion(composition:str) -> list[tuple[str, str]]:
    pattern = r'([A-Z][a-z]*)([0-9]*\.?[0-9]*)'
    matches = re.findall(pattern, composition)

    elements_fractions = []
    for element, fraction in matches:
        frac = float(fraction)  # add logic for missing fractions
        if frac > 1:
            frac = frac / 100.0
        elif frac < 1 and frac > 0:
            frac = frac
        elements_fractions.append((element, frac))
    
    while len(elements_fractions) < 4:
        elements_fractions.append(("", 0.0))
    return elements_fractions
#example usage

# composition = "Sn14.0Bi35.9In50.1"
# elements_fractions = parse_compostion(composition)
# print(elements_fractions)
import math

def calculate_entropy_of_mixing(concentrations):
    pass

def calculate_mix_enthalpy():
    pass
def calculate_eletronegativity_difference():
    pass
def calculate_atomic_radius_difference():
    pass
def calculate_melting_point():
    pass



