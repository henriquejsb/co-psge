import numpy as np
import sge
from sge.parameters import params
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv, _sin_, _cos_

class SRBench():
    def __init__(self, run, invalid_fitness=9999999):
        self.run = run
        self.invalid_fitness = invalid_fitness
        dataset = []
        trn_ind = []
        tst_ind = []
        with open('datasets/dataset_1.csv', 'r') as dataset_file:
            for line in dataset_file.readlines()[1:]:
                dataset.append([float(value.strip(" ")) for value in line.split(",") if value != ""])

        with open('resources/srbench.folds', 'r') as folds_file:
            for _ in range(self.run - 1): folds_file.readline()
            tst_ind = folds_file.readline()
            tst_ind = [int(value.strip(" ")) - 1 for value in tst_ind.split("\t") if value != ""]
            trn_ind = filter(lambda x: x not in tst_ind, range(len(dataset)))
        self.__train_set = []
        self.__train_set_y = []
        for i in trn_ind:
            self.__train_set += [dataset[i][:-1]]
            self.__train_set_y += [dataset[i][-1]]

        self.__train_set = np.asarray(self.__train_set)
        self.__train_set_y = np.asarray(self.__train_set_y)
        self.__test_set = np.asarray([dataset[i][:-1] for i in tst_ind])
        self.__test_set_y = np.asarray([dataset[i][-1] for i in tst_ind])
        
        self.calculate_rrse_denominators()

    def calculate_rrse_denominators(self):
        self.__RRSE_train_denominator = 0
        self.__RRSE_test_denominator = 0
        train_output_mean = float(sum(self.__train_set_y)) / len(self.__train_set_y)
        self.__RRSE_train_denominator = sum([(i - train_output_mean)**2 for i in self.__train_set_y])
        test_output_mean = float(sum(self.__test_set_y)) / len(self.__test_set_y)
        self.__RRSE_test_denominator = sum([(i - test_output_mean)**2 for i in self.__test_set_y])



    def evaluate(self, individual):
        train_mse, train_rrse_a_b, train_mse_a_b = self.calculate_linear_scaling(individual, self.__RRSE_train_denominator, self.__train_set, self.__train_set_y)
        test_mse, test_rrse_a_b, test_mse_a_b = self.calculate_linear_scaling(individual, self.__RRSE_test_denominator, self.__test_set, self.__test_set_y)

        return train_rrse_a_b, {'mse' : train_mse, 'rrse' : train_rrse_a_b,'mse_a_b' : train_mse_a_b, 'test_rrse': test_rrse_a_b}

    def calculate_linear_scaling(self, individual, denominator, dataset, dataset_y):
        try:
            #print(individual)
            code = compile('result = lambda x: ' + individual, 'solution', 'exec')
            globals_code = {'_div_' : _div_, '_log_' : _log_, '_exp_' : _exp_, '_inv_' : _inv_, '_sqrt_':_sqrt_,'_sin_' : _sin_, '_cos_' : _cos_}
            locals_code = {}
            exec(code, globals_code, locals_code)
            func = locals_code['result']
            self.outputs = np.apply_along_axis(func, 1, dataset)
            cov_matrix = np.cov(np.vstack([dataset_y, self.outputs]))
            covariance_y_o = cov_matrix[0,1]
            variance_o = cov_matrix[1,1]
            if not np.isnan(variance_o):
                mse_a_b = self.invalid_fitness
                b = 0
                if variance_o != 0:
                    b = covariance_y_o / variance_o
                a = dataset_y.mean() - b * self.outputs.mean()
                mse = np.sqrt(np.mean(np.square(self.outputs - dataset_y)))
                rrse_a_b = np.sqrt( np.sum(np.square(dataset_y - (a + b * self.outputs))) / denominator)
                mse_a_b = np.mean(np.square(dataset_y - (a + b * self.outputs)))
            else:
                rrse_a_b = mse = mse_a_b = self.invalid_fitness
                b = a = None
            
        except (OverflowError, ValueError) as e:
            rrse_a_b = mse_a_b = mse = self.invalid_fitness
            b = a = None
        return mse, rrse_a_b, mse_a_b

if __name__ == "__main__":
    import sge
    sge.setup("parameters/standard.yml")
    fitness = SRBench(params['RUN'])
    sge.evolutionary_algorithm(evaluation_function=fitness, parameters_file="parameters/standard.yml")