from pathlib import Path
import json
from sympy.core.sympify import sympify
PATH = 'dumps/'
PATTERN = 'best.json'
OUTPUT_FILE = 'dataset_1_best_model'
globals_code = {'|_div_|' : '/', '_log_' : 'log', '_exp_' : 'exp', '_inv_' : '1/', '_sqrt_':'sqrt','_sin_' : 'sin', '_cos_' : 'cos', 'x[0]': 'x0', 'x[1]':'x1'}

def collect_best_files():
    return list(Path(PATH).rglob(PATTERN))

def get_best_ind():
    files = collect_best_files()
    best_fitness = 1000000
    best_ind = None
    for file in files:
        with open(file,'r') as f:
            ind = json.load(f)
            fitness, phenotype = ind['fitness'], ind['phenotype']
            if fitness < best_fitness:
                best_fitness = fitness
                best_ind = phenotype
    return best_ind

def convert_to_sympy(best_ind):
    print(best_ind)
    for operator in globals_code:
        best_ind = best_ind.replace(operator, globals_code[operator])
    print(best_ind)
    sympify(best_ind)
    with open(OUTPUT_FILE,'w') as f:
        f.write(best_ind)

def main():
    best_ind = get_best_ind()
    convert_to_sympy(best_ind)
if __name__ == '__main__':
    main()





