import random
import sys
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
from tqdm import tqdm
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate, mutate_level
from sge.operators.selection import tournament
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)


def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth, 'grammar': grammar.get_dict(), 'mutation_prob':grammar.get_mutation_prob() }


def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()


def evaluate(ind, eval_func):
    mapping_values = [0 for i in ind['genotype']]
    # the grammar of the individual is used in the mapping
    phen, tree_depth = grammar.mapping(ind['genotype'], ind['grammar'], mapping_values)
    quality, other_info = eval_func.evaluate(phen)
    ind['phenotype'] = phen
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth


def setup(parameters_file_path = None):
    if parameters_file_path is not None:
        load_parameters(file_name=parameters_file_path)
    set_parameters(sys.argv[1:])
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    params['EXPERIMENT_NAME'] += "/" + str(params['PROB_MUTATION_GRAMMAR'] * 100) + "/" + str(params['NORMAL_DIST_SD'])
    
    logger.prepare_dumps()
    random.seed(params['SEED'])
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])

def sumProbs(prods, p, diff):
    for prod in prods:
        if prod in p:
            prod[1] = prod[1] + diff


def mutationGrammar(ind):
    gram = ind['grammar']
    for _, prods in gram.items():
        for prod in prods:
            # mutation based on normal distribution
            if random.random() < params['PROB_MUTATION_GRAMMAR']:
                old_prob = prod[1]
                gauss = random.gauss(0.0,params['NORMAL_DIST_SD'])
                new_prob = old_prob + gauss
                diff = (gauss / (len(prods) - 1))
                if new_prob >= 1:
                    for p in prods:
                        p[1] = 0
                        if p == prod:
                            p[1] = 1
                elif new_prob < 0:
                    diff = (old_prob / (len(prods) - 1))
                    for p in prods:
                        p[1] = p[1] + diff
                        if p == prod:
                            p[1] = 0
                else:
                    for p in prods:
                        p[1] = p[1] - diff
                        if p == prod:
                            p[1] = new_prob

                # probabilities check
                # probabilitiesCheck(prods)
                break

    return ind

def mutation_prob_mutation(ind):
    gram = ind['mutation_prob']
    new_p = []
    for p in gram:
        if random.random() < params['PROB_MUTATION_PROBS']:
            gauss = random.gauss(0.0,params['GAUSS_SD'])
            # TODO: no futuro criar bounds
            p = max(p+gauss,0)
            p = min(p,1)
        new_p.append(p)
    ind['mutation_prob'] = new_p
    return ind

def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    setup(parameters_file_path=parameters_file)
    population = list(make_initial_population())
    it = 0

    while it <= params['GENERATIONS']:  
        for i in tqdm(population):
            if i['fitness'] is None:
                evaluate(i, evaluation_function)      
        population.sort(key=lambda x: x['fitness'])

        # logger saves the grammar of the best individual
        
        logger.evolution_progress(it, population)

        new_population = population[:params['ELITISM']]
        while len(new_population) < params['POPSIZE'] - params['ELITISM']:
            if random.random() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            
            # ni = mutationGrammar(ni)
            ni = mutation_prob_mutation(ni)
            # ni = mutate(ni, params['PROB_MUTATION'])
            ni = mutate_level(ni)
            new_population.append(ni)

        population = new_population
        it += 1

