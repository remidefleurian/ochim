# -*- coding: utf-8 -*-

# LIBRARIES ------------------------------------------------------------------
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sed_eval as se
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# RECOMBINE PREPROCESSED DATA ------------------------------------------------
def agg_data():

    files = ['data/preprocessed/k1.csv',
             'data/preprocessed/k2.csv',
             'data/preprocessed/k3.csv',
             'data/preprocessed/k4.csv',
             'data/preprocessed/k5.csv']
             
    inputs = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    inputs.to_csv(f'data/preprocessed/all.csv', index=None)

# GET SAMPLE AS NUMPY ARRAYS -------------------------------------------------
def read_samples(step, k):
    
    print(f'- reading samples ({step})')
    
    if step == 'train':
        k_folds = np.arange(1, 6).take(range(k - 1, k + 2), mode='wrap')
        samples = (pd.read_csv('data/preprocessed/all.csv')
               .query('fold in @k_folds')
               .dropna())
        
    elif step == 'valid':
        k_fold = np.arange(1, 6).take((k + 2), mode='wrap')
        samples = (pd.read_csv('data/preprocessed/all.csv')
               .query('fold == @k_fold')
               .dropna())
        
    elif step == 'test':
        k_fold = np.arange(1, 6).take((k + 3), mode='wrap')
        samples = (pd.read_csv('data/preprocessed/all.csv')
               .query('fold == @k_fold')
               .dropna())
    
    X = samples.to_numpy()[:,4:]
    y = samples.to_numpy()[:,3].astype('int')
    
    return X, y

# BUILD LINEAR SVC -----------------------------------------------------------
def build_svc(X, y, c):
    
    print('- training SVC') 
    
    svm = LinearSVC(dual=False, tol=1e-5, C=c, class_weight='balanced')
    clf = CalibratedClassifierCV(svm)             
    clf.fit(X, y)

    return clf

# GET PROBABILITIES USING PLATT SCALING --------------------------------------
def get_prob(X, svc, k):

    print('- computing probabilities')

    # get track IDs and time
    k_fold = np.arange(1, 6).take((k + 2), mode='wrap')
    preds = (pd.read_csv('data/preprocessed/all.csv')
             .query('fold == @k_fold')
             .dropna()
             .loc[:, ['track_id', 'time']])
    
    # join predictions
    preds['prob'] = svc.predict_proba(X)[:,1]
    
    # join to track IDs and time, replacing NAs by 0
    tracks = (pd.read_csv('data/preprocessed/all.csv')
             .query('fold == @k_fold')
             .loc[:, ['track_id', 'time', 'label']]
             .merge(preds, on=['track_id', 'time'], how='left')
             .fillna(0))
    
    return tracks

# EVALUATE MODELS USING SEGMENT-BASED METRICS --------------------------------
def eval_model(preds, f, k):

    print("- computing evaluation metrics (slow)")

    # format ground truth as required by sed_eval library
    ref = (preds
           .query('label == 1')
           .rename(columns={'time': 'onset'})
           .assign(offset = lambda x: round((x.onset + f/1000), 2))
           .assign(event_label = 'chills')
           .loc[:, ['track_id', 'onset', 'offset', 'event_label']]
           .set_index('track_id')
           .apply(pd.Series.to_dict, axis=1)
           .groupby(level=0)
           .apply(list)
           .rename('ref'))
    
    # store duration of each track
    dur = ((preds.groupby('track_id')['time'].max() - 
            preds.groupby('track_id')['time'].min())
       .rename('dur'))
    
    # sort and sample probs for AUC estimation
    probs = preds['prob'].quantile(np.arange(0, 1.01, 0.01))
    
    # initialise metrics
    perfs = pd.DataFrame(columns=[
        'k_fold', 'threshold', 'auc', 'f', 'f_beta', 'tp_rate', 'bal_acc', 
        'fp_rate', 'precision', 'recall', 'tp', 'tn', 'fp', 'fn']) 
    
    # for each classification threshold
    for i in probs:
                
        # format predictions as required by sed_eval
        est = (preds
         .query('prob > @i')
         .rename(columns={'time': 'onset'})
         .assign(offset = lambda x: round((x.onset + f/1000), 2))
         .assign(event_label = 'chills')
         .loc[:, ['track_id', 'onset', 'offset', 'event_label']]
         .set_index('track_id')
         .apply(pd.Series.to_dict, axis=1)
         .groupby(level=0)
         .apply(list)
         .rename('est')
         )
    
        # combine all data as required by sed_eval
        events = (pd.concat([dur, ref, est], axis=1)
               .apply(pd.Series.to_dict, axis=1))
    
        # initialise sed_eval
        segment_based_metrics = se.sound_event.SegmentBasedMetrics(
            event_label_list=['chills'],
            time_resolution=5.0)
    
        # update metrics for each track
        for j in range(len(events)):
                        
            # convert NaNs to correct format if track has no predictions
            if type(events[j]['est']) == float and pd.isna(events[j]['est']):
                events[j]['est'] = [{}]
            
            # update metrics
            segment_based_metrics.evaluate(
                reference_event_list=events[j]['ref'],
                estimated_event_list=events[j]['est'],
                evaluated_length_seconds=events[j]['dur'])
            
        # extract metrics
        seg = segment_based_metrics.results_overall_metrics()
    
        # save threshold and metrics
        perfs = perfs.append({
            'k_fold': k,
            'threshold': i,
            'f': seg['f_measure']['f_measure'],
            'f_beta': (1+4) * 
                seg['f_measure']['precision'] * 
                seg['f_measure']['recall'] / 
                (4 * 
                    seg['f_measure']['precision'] + 
                    seg['f_measure']['recall']),
            'tp_rate': seg['accuracy']['sensitivity'],
            'bal_acc': seg['accuracy']['balanced_accuracy'],
            'fp_rate': 1 - seg['accuracy']['specificity'],
            'precision': seg['f_measure']['precision'],
            'recall': seg['f_measure']['recall'],
            'tp': segment_based_metrics.class_wise['chills']['Ntp'],
            'tn': segment_based_metrics.class_wise['chills']['Ntn'], 
            'fp': segment_based_metrics.class_wise['chills']['Nfp'], 
            'fn': segment_based_metrics.class_wise['chills']['Nfn']},
            ignore_index=True)
            
    # calculate AUC over all classification thresholds
    auc = metrics.auc(perfs['fp_rate'], perfs['tp_rate'])
    perfs['auc'] = auc
    
    return perfs

# EXPORT AGGREGATED RESULTS --------------------------------------------------
def agg_results():

    path = Path('output/test/eval')
    files = path.glob('*.csv')
    
    # read evaluations for all k-folds at all thresholds
    evals = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    
    # aggregate results for threshold with the best F-beta for each k-fold
    results = (evals
        .sort_values('f_beta', ascending=False)
        .drop_duplicates(subset=['k_fold'])
        .drop(columns='k_fold')
        .mean()
        .round(3)
        .loc[['auc', 'f_beta', 'f', 'precision', 'recall', 'bal_acc']])

    # print results
    print('done!')
    for label, value in results.items():
        print(f'- {label:<10} {value:.3f}')

    return results
