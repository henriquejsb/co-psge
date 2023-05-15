import numpy as np
from sge.parameters import params
import json
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def evolution_progress(generation, pop):
    fitness_samples = [i['fitness'] for i in pop]
    data = '%4d\t%.6e\t%.6e\t%.6e\t%.6e' % (generation, np.min(fitness_samples), np.mean(fitness_samples), np.std(fitness_samples), np.mean([i['other_info']['test_rrse'] for i in pop]))
    if params['VERBOSE']:
        print(data)
    save_progress_to_file(data, pop[0],generation)
    save_best(pop)

def save_progress_to_file(data, best, generation):
    with open('%s/run_%d/progress_report.csv' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data + '\n')
    c = json.dumps(best, cls=NumpyEncoder)
    open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'a').write(c)


def save_step(generation, population):
    c = json.dumps(population, cls=NumpyEncoder)
    open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'a').write(c)

def save_best(population):
    c = json.dumps(population[0],cls=NumpyEncoder)
    open('%s/run_%d/best.json' % (params['EXPERIMENT_NAME'], params['RUN']), 'w').write(c)


def save_parameters():
    params_lower = dict((k.lower(), v) for k, v in params.items())
    c = json.dumps(params_lower)
    open('%s/run_%d/parameters.json' % (params['EXPERIMENT_NAME'], params['RUN']), 'a').write(c)


def prepare_dumps():
    try:
        os.makedirs('%s/run_%d' % (params['EXPERIMENT_NAME'], params['RUN']))
    except FileExistsError as e:
        pass
    save_parameters()