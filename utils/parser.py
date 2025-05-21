import re 

def parse_compostion(composition: str) -> list[tuple[str, float]]:
    pattern = r'([A-Z][a-z]*)([0-9]*\.?[0-9]*)'
    matches = re.findall(pattern, composition)

    elements_fractions = []
    for element, fraction in matches:
        # print(f"Element: {element}, Fraction: {fraction}")
        if fraction == "":
            fraction = "1"
        frac = float(fraction)
        if frac > 1:
            frac = frac / 100.0
        elements_fractions.append((element, frac))
    
    # Normalize fractions to ensure they sum to 1
    total = sum(frac for _, frac in elements_fractions)
    if total != 1.0:
        elements_fractions = [(element, frac/total) for element, frac in elements_fractions]
    
    while len(elements_fractions) < 4:
        elements_fractions.append(("", 0.0))
    return elements_fractions