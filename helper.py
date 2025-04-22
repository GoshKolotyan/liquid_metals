import re 

def parse_compostion(composition:str) -> list[tuple[str, str]]:
    pattern = r'([A-Z][a-z]*)([0-9]*\.?[0-9]*)'
    matches = re.findall(pattern, composition)

    elements_fractions = []
    for element, fraction in matches:
        frac = float(fraction)  # add logic for missing fractions
        elements_fractions.append((element, frac))
    return elements_fractions

#example usage

composition = "Sn14.0Bi35.9In50.1"
elements_fractions = parse_compostion(composition)
print(elements_fractions)