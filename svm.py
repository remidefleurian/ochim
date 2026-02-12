# -*- coding: utf-8 -*-

# LIBRARIES ------------------------------------------------------------------
import sys
import pandas as pd
from src.svm_helpers import *

# MAIN FUNCTION --------------------------------------------------------------
def run_svm(c, f):
    
    # recombine preprocessed data
    print('recombining input data')
    agg_data()

    # for each k-fold
    for k in range(1, 6):
        
        # print progress
        samples = np.arange(1, 6).take(range(k - 1, k + 2), mode='wrap')
        k_train = ", ".join(str(x) for x in samples)
        k_valid = np.arange(1, 6).take((k + 2), mode='wrap')
        k_test = np.arange(1, 6).take((k + 3), mode='wrap')
        print(f'training on k = {k_train},',
              f'validating on k =  {k_valid},',
              f'testing on k = {k_test}')
        
        # read training samples
        X_train, y_train = read_samples('train', k)
        
        # build model
        svc = build_svc(X_train, y_train, c)
        
        # read validation samples
        X_valid, y_valid = read_samples('valid', k)
        
        # get and save predictions
        preds = get_prob(X_valid, svc, k)
        preds.to_csv(f'output/validate/preds/preds-k{k}.csv', index=None)
        
        # get and save evaluation metrics
        evals = eval_model(preds, f, k)
        evals.to_csv(f'output/validate/eval/eval-k{k}.csv', index=None)

        # read testing samples
        X_test, y_test = read_samples('test', k)
        
        # get and save predictions
        preds = get_prob(X_test, svc, k + 1)
        preds.to_csv(f'output/test/preds/preds-k{k}.csv', index=None)
        
        # get and save evaluation metrics
        evals = eval_model(preds, f, k)
        evals.to_csv(f'output/test/eval/eval-k{k}.csv', index=None)
        
    # aggregate results across k-folds
    results = agg_results()
    results.to_frame().T.to_csv('output/results/results.csv', index=None)

# COMMAND LINE ---------------------------------------------------------------
if __name__ == '__main__':
    
    # best parameters from validation run after grid search
    c = 1e-6 # regularization
    f = 500 # input frame size
    
    run_svm(c, f)
