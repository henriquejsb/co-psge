import random

SEED = 2225
TOTAL = 2000
TEST = 0.05
N_FOLDS = 100

random.seed(SEED)

for i in range(N_FOLDS):
    data = []
    while len(data) != int(TOTAL*TEST):
        sample =  str(random.randint(0,TOTAL-1))
        if sample in data:
            continue
        data += [sample]
    print('\t'.join(data))
