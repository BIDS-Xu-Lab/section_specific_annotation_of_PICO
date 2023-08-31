# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:19:05 2020

@author: jli34
"""

import os
import copy
import numpy as np
import pandas as pd
import pickle
from collections import Counter, OrderedDict
from evaluate import evaluate
# i2b2_n2c2_synth_dataset = 'i2b2_n2c2_synth'
# i2b2_hpi_dataset = 'i2b2_hpi' 
# mtsamples_hpi_dataset = 'mtsamples_hpi'
# utnotes_hpi_dataset = 'utnotes_hpi'
# i2b2_n2c2_synth_for_i2b2_hpi_dataset = 'i2b2_n2c2_synth_for_i2b2_hpi'
# i2b2_n2c2_synth_for_mtsamples_hpi_dataset = 'i2b2_n2c2_synth_for_mtsamples_hpi'
# i2b2_n2c2_synth_for_utnotes_hpi_dataset = 'i2b2_n2c2_synth_for_utnotes_hpi'

results_index = {
    'clamp_eval': {
        'ner_tags': {
            'exact precision': 0, 'exact recall': 1, 'exact fb1': 2, 
            'relax precision': 3, 'relax recall': 4, 'relax fb1': 5, 
            'right': 6, 'right_predict': 7, 'right_gold': 8, 
            'predict': 9, 'gold': 10,
        },
    },
    'clamp_tfner_eval' : {
        'ner_tags': {
            'right': 1, 'predict': 2, 'gold': 3, 
            'precision': 4, 'recall': 5, 'fb1': 6, 
        },
    },   
    'conlleval': {
        'processed': {
            'tokens': 1, 'phrases': 4, 
            'found': 7, 'correct': 10,
        },
        'overall': {
            'accuracy': 1, 'precision': 3, 
            'recall': 5, 'fb1': 7,
        }, 
        'ner_tags': {
            'precision': 2, 'recall': 4, 
            'fb1': 6, 'found': 7,
        },
    },     
}



def get_test_results(results, task):
    eval_type = task['eval_type']
    if eval_type=='clamp_tfner_eval':
        return (get_test_results_clamp_tfner_eval(results, task))
    elif eval_type=='clamp_eval':
        return (get_test_results_clamp_eval(results, task))
    else: # eval_type=='colleval':
        return (get_test_results_conlleval(results, task))

def get_test_results_clamp_tfner_eval(results, task):
    test_results =copy.deepcopy(task['ner_results_dict'])
    result_index = task['result_index']
    ner_tags = task['ner_tags']
    if len(results) <= 0:
        return test_results
    # I2B2 results:
    '''Relax
    semantic	right	predict	gold	Precision	Recall	F1
    drug	1552	1635	1686	0.949	0.921	0.935
    problem	4692	5016	5079	0.935	0.924	0.930
    test	2832	3053	3157	0.928	0.897	0.912
    treatment	1014	1227	1372	0.826	0.739	0.780
    Exact
    semantic	right	predict	gold	Precision	Recall	F1
    drug	1487	1635	1686	0.909	0.882	0.896
    problem	4241	5016	5079	0.845	0.835	0.840
    test	2603	3053	3157	0.853	0.825	0.838
    treatment	869	1227	1372	0.708	0.633	0.669    processed = results[0].split()
    '''   
    #overall = results[1].split()
    #test_result['overall'] = \
    #                    {'accuracy': float(overall[result_index['overall']['accuracy']].split('%')[0]), \
    #                    'precision': float(overall[result_index['overall']['precision']].split('%')[0]), \
    #                    'recall': float(overall[result_index['overall']['recall']].split('%')[0]), \
    #                    'fb1': float(overall[result_index['overall']['fb1']]),}    
    ##NOTE: SUPPOSED THE ORIGINAL RESULTS ARE IN THE SAME ORDER AS THE ORDER OF NER-TAGS!!!
    ## A BETTER WAY IS TO FIND EACH NER-TAGS IN RESULTS
    result_type = ''
    for idx in range(len(results)):        
        result = results[idx].split()
        if not result:
            continue
        elif result[0] == 'Exact':
            result_type = 'exact'
            continue        
        elif result[0] == 'Relax':
            result_type = 'relax'
            continue
        elif result[0] == 'semantic':
            continue
        tag = result[0]
        if tag in ner_tags:
            precision = float('{:.3f}'.format(float(result[result_index['ner_tags']['precision']])))            
            recall = float('{:.3f}'.format(float(result[result_index['ner_tags']['recall']])))
            fb1 = float('{:.3f}'.format(float(result[result_index['ner_tags']['fb1']])))
            right = int(result[result_index['ner_tags']['right']])
            predict = int(result[result_index['ner_tags']['predict']])            
            gold = int(result[result_index['ner_tags']['gold']])
            fp = predict - right
            fn = gold - right
            tn = 0
            test_results[result_type][tag] = \
                        {'precision': precision, \
                        'recall': recall, \
                        'fb1': fb1, \
                        'TP': right, \
                        'FP': fp, \
                        'FN': fn, \
                        'TN': tn, \
                        }
    # calculate overall for relax and exact
    for rt in test_results:   
        # the results contain certain tags which not included in ner-tags, so re-calculate the overall
        rights = 0
        predicts = 0
        golds = 0
        #tns = 0
        #tpfpfntns = 0
        precision_sum = 0.0
        recall_sum = 0.0
        for tag in ner_tags:
            precision = test_results[rt][tag]['precision']            
            recall = test_results[rt][tag]['recall']
            #accuracy_sum += (right+tn)/(predict + gold - right + tn)
            precision_sum += precision
            recall_sum += recall
            right = test_results[rt][tag]['right']
            predict = test_results[rt][tag]['predict']
            gold = test_results[rt][tag]['gold']
            rights += right
            predicts += predict
            golds += gold
        #test_result['overall']['accuracy'] = 0.0 # no True-Negative number
        precision_micro = float('{:.3f}'.format(rights / predicts))
        test_results[rt]['micro_overall']['precision'] = precision_micro
        recall_micro = float('{:.3f}'.format(rights / golds))
        test_results[rt]['micro_overall']['recall'] = recall_micro
        test_results[rt]['micro_overall']['fb1'] = float('{:.3f}'.format(2.0*precision_micro*recall_micro / (precision_micro+recall_micro)))
        test_results[rt]['micro_overall']['TP'] = rights
        test_results[rt]['micro_overall']['FP'] = predicts - rights
        test_results[rt]['micro_overall']['FN'] = golds - rights
        test_results[rt]['micro_overall']['TN'] = 0
        precision_macro = float('{:.3f}'.format(precision_sum / len(ner_tags)))
        test_results[rt]['macro_overall']['precision'] = precision_macro
        recall_macro = float('{:.3f}'.format(recall_sum / len(ner_tags)))
        test_results[rt]['macro_overall']['recall'] = recall_macro
        test_results[rt]['macro_overall']['fb1'] = float('{:.3f}'.format(2.0*precision_macro*recall_macro / (precision_macro+recall_macro)))
        test_results[rt]['macro_overall']['TP'] = rights
        test_results[rt]['macro_overall']['FP'] = predicts - rights
        test_results[rt]['macro_overall']['FN'] = golds - rights
        test_results[rt]['macro_overall']['TN'] = 0
          
    return test_results

def get_test_results_clamp_eval(results, task):
    test_results =copy.deepcopy(task['ner_results_dict'])
    result_index = task['result_index']
    ner_tags = task['ner_tags']
    if len(results) <= 0:
        return test_results
    # ctg results:
    '''P(exact)	R(exact)	F1(exact)	P(relax)	R(relax)	F1(relax)	right	right_predict	right_gold	predict	gold	Semantic
       0.812	0.799	0.805	0.928	0.907	0.917	496	563	567	611	621	problem
       0.820	0.795	0.807	0.896	0.864	0.880	205	223	224	250	258	test
       0.748	0.671	0.707	0.839	0.746	0.789	116	129	130	155	173	treatment
       0.804	0.777	0.790	0.906	0.870	0.888	817	915	921	1016	1052	overall
    '''   
    ##NOTE: SUPPOSED THE ORIGINAL RESULTS ARE IN THE SAME ORDER AS THE ORDER OF NER-TAGS!!!
    ## A BETTER WAY IS TO FIND EACH NER-TAGS IN RESULTS    
    exact_precision_sum = 0
    exact_recall_sum = 0
    relax_precision_sum = 0
    relax_recall_sum = 0        
    for idx in range(len(results)):        
        result = results[idx].split()
        if idx == 0 or not result:
            # ignore the first title line
            continue        
        exact_precision = float('{:.3f}'.format(float(result[result_index['ner_tags']['exact precision']])))
        exact_recall = float('{:.3f}'.format(float(result[result_index['ner_tags']['exact recall']])))
        exact_fb1 = float('{:.3f}'.format(float(result[result_index['ner_tags']['exact fb1']])))
        relax_precision = float('{:.3f}'.format(float(result[result_index['ner_tags']['relax precision']])))
        relax_recall = float('{:.3f}'.format(float(result[result_index['ner_tags']['relax recall']])))
        relax_fb1 = float('{:.3f}'.format(float(result[result_index['ner_tags']['relax fb1']])))
        right = int(result[result_index['ner_tags']['right']])
        right_predict = int(result[result_index['ner_tags']['right_predict']])
        right_gold = int(result[result_index['ner_tags']['right_gold']])
        predict = int(result[result_index['ner_tags']['predict']])
        gold = int(result[result_index['ner_tags']['gold']])
        exact_tp = right
        exact_fp = predict - right
        exact_fn = gold - right        
        exact_tn = 0
        relax_tp = right_gold
        relax_fp = predict - right_gold
        relax_fn = gold - right_gold
        relax_tn = 0

        tag = result[-1]
        if tag in ner_tags or tag == 'overall':
            if tag == 'overall':
                new_tag = 'micro_overall'
            else:
                new_tag = tag
                exact_precision_sum += exact_precision
                exact_recall_sum += exact_recall
                relax_precision_sum += relax_precision
                relax_recall_sum += relax_recall
            test_results['exact'][new_tag] = {
                        'precision': exact_precision, 
                        'recall': exact_recall, 
                        'fb1': exact_fb1,
                        'TP': exact_tp,
                        'FP': exact_fp,
                        'FN': exact_fn,
                        'TN': exact_tn,
                        }
            test_results['relax'][new_tag] = {
                        'precision': relax_precision, 
                        'recall': relax_recall,
                        'fb1': relax_fb1,
                        'TP': relax_tp,
                        'FP': relax_fp,
                        'FN': relax_fn,
                        'TN': relax_tn,
                        }
    # calculate macro_overall for relax and exact
    exact_precision = float('{:.3f}'.format(exact_precision_sum/len(ner_tags)))
    exact_recall = float('{:.3f}'.format(exact_recall_sum/len(ner_tags)))
    exact_f1 = float('{:.3f}'.format(2*exact_precision*exact_recall/(exact_precision+exact_recall)))
    exact_tp = test_results['exact']['micro_overall']['TP']
    exact_fp = test_results['exact']['micro_overall']['FP']
    exact_fn = test_results['exact']['micro_overall']['FN']
    exact_tn = test_results['exact']['micro_overall']['TN']
    relax_precision = float('{:.3f}'.format(relax_precision_sum/len(ner_tags)))
    relax_recall = float('{:.3f}'.format(relax_recall_sum/len(ner_tags)))
    relax_f1 = float('{:.3f}'.format(2*relax_precision*relax_recall/(relax_precision+relax_recall)))
    relax_tp = test_results['relax']['micro_overall']['TP']
    relax_fp = test_results['relax']['micro_overall']['FP']
    relax_fn = test_results['relax']['micro_overall']['FN']
    relax_tn = test_results['relax']['micro_overall']['TN']
    test_results['exact']['macro_overall'] = {
                'precision': exact_precision, 
                'recall': exact_recall, 
                'fb1': exact_fb1,
                'TP': exact_tp,
                'FP': exact_tp,
                'FN': exact_fn,
                'TN': exact_tn,
                }                
    test_results['relax']['macro_overall'] = {
                'precision': relax_precision, 
                'recall': relax_recall,
                'fb1': relax_fb1,
                'TP': relax_tp,
                'FP': relax_fp,
                'FN': relax_fn,
                'TN': relax_tn,
                }    
    return test_results

def get_test_results_conlleval(results, task):
    test_results = task['ner_results_dict']
    ner_tags = task['ner_tags']
    result_index = task['result_index']
    #test_result = copy.deepcopy(ner_result_dict)
    if len(results) <= 0:
        return test_results
    # I2B2 results:
    #processed 52526 tokens with 6366 phrases; found: 6236 phrases; correct: 5396.   
    #accuracy:  95.26%; precision:  86.53%; recall:  84.76%; FB1:  85.64
    #      problem: precision:  84.56%; recall:  84.86%; FB1:  84.71  2532
    #         test: precision:  89.54%; recall:  86.48%; FB1:  87.98  2036
    #    treatment: precision:  85.85%; recall:  82.54%; FB1:  84.16  1668
    # SemEval results:
    #processed 58603 tokens with 3266 phrases; found: 3306 phrases; correct: 2460.
    #accuracy:  96.88%; precision:  74.41%; recall:  75.32%; FB1:  74.86
    #            BL: precision:  69.58%; recall:  73.91%; FB1:  71.68  802
    #            CC: precision:  65.52%; recall:  61.79%; FB1:  63.60  116
    #            CO: precision:  73.08%; recall:  33.93%; FB1:  46.34  26
    #        DISORDER: precision:  79.16%; recall:  79.94%; FB1:  79.55  1867
    #            GC: precision:  57.14%; recall:  19.05%; FB1:  28.57  7
    #            NI: precision:  71.71%; recall:  79.40%; FB1:  75.36  258
    #            SC: precision:  78.95%; recall:  62.50%; FB1:  69.77  19
    #            SV: precision:  77.00%; recall:  81.91%; FB1:  79.38  100
    #            UI: precision:  43.24%; recall:  43.24%; FB1:  43.24  111
    processed = results[0].split()
    processed_tokens = int(processed[result_index['processed']['tokens']])
    overall = results[1].split()
    tag_results = results[2:]
    result_type = 'exact'
    for result in tag_results:
        result = result.split()
        if not result:
            continue        
        tag = result[0][:-1]

        if tag in ner_tags:
            vals = result[result_index['ner_tags']['precision']].split('%')
            val = float(vals[0]) if vals else 0.0
            precision = float('{:.3f}'.format(val/100.0))
            vals = result[result_index['ner_tags']['recall']].split('%')
            val = float(vals[0]) if vals else 0.0
            recall = float('{:.3f}'.format(val/100.0))
            vals = result[result_index['ner_tags']['fb1']].split('%')
            val = float(vals[0]) if vals else 0.0
            fb1 = float('{:.3f}'.format(val/100.0))
            found = int(result[result_index['ner_tags']['found']])
            TP = int(found * precision)
            FP = found - TP
            FN = int(TP/recall - TP)
            TN = processed_tokens - TP - FP - FN
            test_results[result_type][tag] = {
                        'precision': precision, 
                        'recall': recall, 
                        'fb1': fb1, 
                        'TP': TP,
                        'FP': FP, 
                        'FN': FN, 
                        'TN': TN,
                        }
    # calculate overall for exact type
    for rt in ['exact']:   
        # the results contain certain tags which not included in ner-tags, so re-calculate the overall
        rights = 0
        predicts = 0
        golds = 0
        tns = 0
        #tns = 0
        #tpfpfntns = 0
        precision_sum = 0.0
        recall_sum = 0.0
        for tag in ner_tags:
            precision = test_results[rt][tag]['precision']
            recall = test_results[rt][tag]['recall']
            #accuracy_sum += (right+tn)/(predict + gold - right + tn)
            precision_sum += precision
            recall_sum += recall
            right = test_results[rt][tag]['TP']
            predict = test_results[rt][tag]['TP'] + test_results[rt][tag]['FP']
            gold = test_results[rt][tag]['TP'] + test_results[rt][tag]['FN']
            tn = test_results[rt][tag]['TN']
            rights += right
            predicts += predict
            golds += gold  
            tns += tn
        precision_micro = float('{:.3f}'.format(rights / predicts))
        test_results[rt]['micro_overall']['precision'] = precision_micro
        recall_micro = float('{:.3f}'.format(rights / golds))
        test_results[rt]['micro_overall']['recall'] = recall_micro
        test_results[rt]['micro_overall']['fb1'] = float('{:.3f}'.format(2.0*precision_micro*recall_micro / (precision_micro+recall_micro)))
        test_results[rt]['micro_overall']['TP'] = rights
        test_results[rt]['micro_overall']['FP'] = predicts - rights
        test_results[rt]['micro_overall']['FN'] = golds - rights
        test_results[rt]['micro_overall']['TN'] = 0
        precision_macro = float('{:.3f}'.format(precision_sum / len(ner_tags)))
        test_results[rt]['macro_overall']['precision'] = precision_macro
        recall_macro = float('{:.3f}'.format(recall_sum / len(ner_tags)))
        test_results[rt]['macro_overall']['recall'] = recall_macro
        test_results[rt]['macro_overall']['fb1'] = float('{:.3f}'.format(2.0*precision_macro*recall_macro / (precision_macro+recall_macro)))
        test_results[rt]['macro_overall']['TP'] = rights
        test_results[rt]['macro_overall']['FP'] = predicts - rights
        test_results[rt]['macro_overall']['FN'] = golds - rights
        test_results[rt]['macro_overall']['TN'] = tns
        # calculate accuracy
        test_results[rt]['accuracy']['accuracy'] =  float('{:.3f}'.format((rights+tns)/(4*processed_tokens)))
    
    return test_results

def voting_nfold_results(preds_nfold_voting, voting_pred_sep_tag, voting_pred_nfold_sep_tag, output_pred_sep_tag):
    preds_voting = []
    pred_previous = ''
    for idx, pred_nfold in enumerate(preds_nfold_voting):
        #print(pred_nfold)
        if not pred_nfold:
            # new line
            preds_voting.append(pred_nfold)
            pred_previous = ''
        else:
            items = pred_nfold.split(voting_pred_sep_tag)
            text = items[0]
            gold = items[1]
            preds = items[2].split(voting_pred_nfold_sep_tag)
            preds_cnt = Counter(preds)
            pred = preds_cnt.most_common(1)[0][0]
            if pred.startswith('B-'):
                # check whether pred_previous starts with B-/I- and the same label as pred
                if (pred_previous[0:2] in ['B-', 'I-']) \
                   and (pred_previous.split('-')[-1] == pred.split('-')[-1]):
                   pred = 'I-' + pred.split('-')[-1]
                   print(f"Warning: update voting's B- to I- at line {idx+1}: {pred_nfold} -> {pred}")
            elif pred.startswith('I-'):
                # check whether pred_previous starts with B-/I- and the same label as pred
                if not ((pred_previous[0:2] in ['B-', 'I-']) and 
                   (pred_previous.split('-')[-1] == pred.split('-')[-1])):
                    pred = 'B-' + pred.split('-')[-1]
                    print(f"Warning: update voting's I- to B- at line {idx+1}: {pred_nfold} -> {pred}")
            elif pred != 'O':
                print(f'WARNING: unkown labels at line {idx+1}: {pred_nfold}')
            pred_previous = pred
            preds_voting.append(output_pred_sep_tag.join([text, gold, pred]))
    return preds_voting

def vote_combining_nfold_results(
            n_fold, data_root_path, 
            source_result_subdir='output/epoch25/source_fold_{idx}/score', 
            source_voting_predict_fn='testb.preds.txt', 
            voting_result_subdir='output/epoch25/score_voting', 
            voting_all_predict_fn='testb.all.preds.txt', 
            voting_predict_fn='testb.preds.txt', 
            voting_result_fn='score.testb.metrics.txt',
            run_eval_prog='perl',
            eval_path_prog='conlleval',
            source_pred_sep_tag_type='space',
            voting_pred_sep_tag_type='tab',
            voting_pred_nfold_sep_tag_type='space'):
    source_pred_sep_tag = ' ' if source_pred_sep_tag_type=='space' else '\t'
    voting_pred_sep_tag = ' ' if voting_pred_sep_tag_type=='space' else '\t'
    voting_pred_nfold_sep_tag = ' ' if voting_pred_nfold_sep_tag_type=='space' else '\t'
    dest_dir = os.path.join(data_root_path, voting_result_subdir)    
    os.makedirs(dest_dir, exist_ok=True)
    dest_all_predict_pfn = os.path.join(dest_dir, voting_all_predict_fn)
    dest_predict_pfn = os.path.join(dest_dir, voting_predict_fn)
    dest_result_pfn = os.path.join(dest_dir, voting_result_fn)
    preds_nfold = []
    preds_nfold_voting = []
    for idx in range(n_fold):
        source_dir = os.path.join(data_root_path, source_voting_result_subdir.format(idx=idx+1))
        source_pred_pfn = os.path.join(source_dir, source_voting_predict_fn)
        with open(source_pred_pfn, 'r', encoding='utf-8') as rf:
            preds_nfold.append(rf.read().split('\n'))
    for idx2, line_nfold in enumerate(zip(*preds_nfold)):
        #print(pred_nfold)
        pred_voting = ''
        for idx3, line in enumerate(line_nfold):
            if line.strip():
                items = line.split(source_pred_sep_tag)
                text = items[0]
                gold = items[1]
                pred = items[2]
                if not gold:
                    print(f'WARNING: empty gold at line {idx2+1} for source_fold_{idx3+1}')
                    gold = 'O'                
                if not pred:
                    print(f'WARNING: empty prediction for source_fold_{idx2+1}')
                    pred = 'O'
                if idx3 == 0:
                    # source_fold_1
                    pred_voting = voting_pred_sep_tag.join([text, gold, pred])
                else:
                    pred_voting += voting_pred_nfold_sep_tag + pred
            else:
                # new line
                pred_voting = line
                break
        preds_nfold_voting.append(pred_voting)    
    # write all nfold results to dest_all_predict_pfn
    with open(dest_all_predict_pfn, 'w', encoding='utf-8') as wf:
        wf.write('\n'.join(preds_nfold_voting))
    # voting all nfold results into voting_predict_fn
    preds_voting = voting_nfold_results(preds_nfold_voting, voting_pred_sep_tag, voting_pred_nfold_sep_tag, source_pred_sep_tag)
    with open(dest_predict_pfn, 'w', encoding='utf-8') as wf:
        wf.write('\n'.join(preds_voting))    
    
    # evaluate preds_voting results
    #cmd = f'{run_eval_prog} {eval_path_prog} < {dest_predict_pfn} | tee {dest_result_pfn}'
    #print(cmd)
    #os.system(cmd)
    evaluate_test_predict_results(run_eval_prog, eval_path_prog, eval_type, dest_predict_pfn, dest_result_pfn, source_pred_sep_tag_type)

def write_summarized_results_clamp_eval(summarized_results_xls_pfn, n_fold, ner_tags, test_results_dict, test_results, exact_relax_style):
    #ner_n_fold_with_average_summarize = [OrderedDict()] * (n_fold+1)
    ner_n_fold_with_average_summarize = OrderedDict()
    #n_fold_average_xsl_result_pfn = os.path.join(summarized_result_path, '{}_{}_fold_with_average.xlsx'.format(task_name, n_fold))
    print(f'writing {summarized_results_xls_pfn}')
    with pd.ExcelWriter(summarized_results_xls_pfn) as writer:
        for fold_idx in range(n_fold+1):
            if exact_relax_style == 'horizontal_separate':
                # add ner_tags at first
                for tag in ner_tags:
                    ner_n_fold_with_average_summarize[tag] = []        
                # add 'overall' at last 
                ner_n_fold_with_average_summarize['micro_overall'] = []
                ner_n_fold_with_average_summarize['macro_overall'] = []
                # write exact and inexact horizontal_separately
                columns1 = ','.join(["{},~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~".format(col) for col in test_results_dict.keys()]).split(',')
                columns2 = ['exact P', 'exact R', 'exact F1', 'exact TP', 'exact FP', 'exact FN', 'exact TN', 'relax P', 'relax R', 'relax F1', 'relax TP', 'relax FP', 'relax FN', 'relax TN'] * (len(test_results_dict.keys()))
                columns1_2_startcol = 1
                prf1_startcol = 0
            elif exact_relax_style == 'horizontal_combine':
                # add ner_tags at first
                for tag in ner_tags:
                    ner_n_fold_with_average_summarize[tag] = []        
                # add 'overall' at last 
                ner_n_fold_with_average_summarize['micro_overall'] = []
                ner_n_fold_with_average_summarize['macro_overall'] = []
                # write exact and inexact horizontal_combined 
                columns1 = ','.join(["{},~~~,~~~,~~~,~~~,~~~,~~~".format(col) for col in test_results_dict.keys()]).split(',')
                columns2 = ['P', 'R', 'F1', 'TP', 'FP', 'FN', 'TN', ] * (len(test_results_dict.keys()))
                columns1_2_startcol = 1
                prf1_startcol = 0
            elif exact_relax_style == 'vertical':
                # add ner_tags at first
                for tag in ner_tags:
                    ner_n_fold_with_average_summarize[tag+' exact'] = []
                    ner_n_fold_with_average_summarize[tag+' relax'] = []
                # add 'overall' at last 
                ner_n_fold_with_average_summarize['micro_overall exact'] = []
                ner_n_fold_with_average_summarize['micro_overall relax'] = []
                ner_n_fold_with_average_summarize['macro_overall exact'] = []
                ner_n_fold_with_average_summarize['macro_overall relax'] = []
                # write exact and inexact vertically
                columns1 = ','.join(["{},~~~,~~~,~~~,~~~,~~~,~~~".format(col) for col in test_results_dict.keys()]).split(',')
                columns2 = ['P', 'R', 'F1', 'TP', 'FP', 'FN', 'TN'] * (len(test_results_dict.keys()))
                columns1_2_startcol = 2
                prf1_startcol = 1
            columns1 = [item.replace('~~~',' ') if item=='~~~' else item  for item in columns1]
            start_row = 2
            sheet_name = '{}'.format(str(n_fold)+'_folds_average' if fold_idx==n_fold else str(fold_idx+1)+'_fold')
            #sheet_name = sheet_name[-20:] if len(sheet_name) >= 31 else sheet_name #Excel worksheet name must be <= 31 chars
            df = pd.DataFrame(data=columns1).transpose() #index=index,
            df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=columns1_2_startcol, index=False, header=False)
            start_row += len(df)
            df = pd.DataFrame(data=columns2).transpose() #index=index,
            df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=columns1_2_startcol, index=False, header=False)
            start_row += len(df)
            for res_idx, res in enumerate(ner_result_dict.keys()):
                if res == 'processed' or res == 'accuracy':
                    #ignore 'processed'
                    continue
                for dt_idx, dt in enumerate(test_results_dict.keys()):
                    #if res in ['micro_overall', 'macro_overall']:
                    # the first [0:3]-index item of overall items are be P/R/F1
                    prf1_indexes = {'start': 0,'end': 3}
                    rt_idx_exact = 0 # exact
                    prf1_exact = ['{:.03f}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx_exact][res_idx][prf1_indexes['start']:prf1_indexes['end']]] #P,R,F1
                    tpfpfntn_indexes = {'start': 3,'end': 7}
                    tpfpfntn_exact = ['{}'.format(int(item)) for item in test_results_np[fold_idx][dt_idx][rt_idx_exact][res_idx][tpfpfntn_indexes['start']:tpfpfntn_indexes['end']]] #TP, FP, FN, TN
                    rt_idx_relax = 1 # relax
                    prf1_relax = ['{:.03f}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx_relax][res_idx][prf1_indexes['start']:prf1_indexes['end']]] #P,R,F1
                    tpfpfntn_relax = ['{}'.format(int(item)) for item in test_results_np[fold_idx][dt_idx][rt_idx_relax][res_idx][tpfpfntn_indexes['start']:tpfpfntn_indexes['end']]] #TP, FP, FN, TN
                    if exact_relax_style == 'horizontal_separate':
                        ner_n_fold_with_average_summarize[res].extend(prf1_exact)
                        ner_n_fold_with_average_summarize[res].extend(tpfpfntn_exact)
                        ner_n_fold_with_average_summarize[res].extend(prf1_relax)
                        ner_n_fold_with_average_summarize[res].extend(tpfpfntn_relax)
                    elif exact_relax_style == 'horizontal_combine':                        
                        prf1_exact_relax = []
                        tpfpfntn_exact_relax = []
                        for (exact, relax) in zip(prf1_exact, prf1_relax):
                            prf1_exact_relax.append('{}({})'.format(exact, relax))
                        for (exact, relax) in zip(tpfpfntn_exact, tpfpfntn_relax):
                            tpfpfntn_exact_relax.append('{}({})'.format(exact, relax))
                        ner_n_fold_with_average_summarize[res].extend(prf1_exact_relax)
                        ner_n_fold_with_average_summarize[res].extend(tpfpfntn_exact_relax)
                    elif exact_relax_style == 'vertical':
                        ner_n_fold_with_average_summarize[res+' exact'].extend(prf1_exact)
                        ner_n_fold_with_average_summarize[res+' exact'].extend(tpfpfntn_exact)
                        ner_n_fold_with_average_summarize[res+' relax'].extend(prf1_relax)
                        ner_n_fold_with_average_summarize[res+' relax'].extend(tpfpfntn_relax)
            #index = ner_tags 
            #index.append('overall') 
            df = pd.DataFrame(data=ner_n_fold_with_average_summarize).transpose() #index=index,
            #df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=index, columns=columns2 )
            df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=prf1_startcol, header=False) #, index=index)
            start_row += len(df) + 1

def write_summarized_results_clamp_dockersimple(summarized_results_xls_pfn, n_fold, ner_tags, test_results_dict, test_results_np, exact_relax_style='horizontal_separate'):
    #ner_n_fold_with_average_summarize = [OrderedDict()] * (n_fold+1)
    ner_n_fold_with_average_summarize = OrderedDict()
    #n_fold_average_xsl_result_pfn = os.path.join(summarized_result_path, '{}_{}_fold_with_average.xlsx'.format(task_name, n_fold))
    print(f'writing {summarized_results_xls_pfn}')
    with pd.ExcelWriter(summarized_results_xls_pfn) as writer:
        for fold_idx in range(n_fold+1):
            if exact_relax_style == 'horizontal_separate':
                # add ner_tags at first
                for tag in ner_tags:
                    ner_n_fold_with_average_summarize[tag] = []        
                # add 'overall' at last 
                ner_n_fold_with_average_summarize['micro_overall'] = []
                ner_n_fold_with_average_summarize['macro_overall'] = []
                # write exact and inexact horizontal_separately
                columns1 = ','.join(["{},~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~".format(col) for col in test_results_dict.keys()]).split(',')
                columns2 = ['exact P', 'exact R', 'exact F1', 'exact TP', 'exact FP', 'exact FN', 'relax P', 'relax R', 'relax F1', 'relax TP', 'relax FP', 'relax FN'] * (len(test_results_dict.keys()))
                columns1_2_startcol = 1
                prf1_startcol = 0
            elif exact_relax_style == 'vertical':
                # add ner_tags at first
                for tag in ner_tags:
                    ner_n_fold_with_average_summarize[tag+' exact'] = []
                    ner_n_fold_with_average_summarize[tag+' relax'] = []
                # add 'overall' at last 
                ner_n_fold_with_average_summarize['micro_overall exact'] = []
                ner_n_fold_with_average_summarize['micro_overall relax'] = []
                ner_n_fold_with_average_summarize['macro_overall exact'] = []
                ner_n_fold_with_average_summarize['macro_overall relax'] = []
                # write exact and inexact vertically
                columns1 = ','.join(["{},~~~,~~~,~~~,~~~,~~~".format(col) for col in test_results_dict.keys()]).split(',')
                columns2 = ['P', 'R', 'F1', 'TP', 'FP', 'FN'] * (len(test_results_dict.keys()))
                columns1_2_startcol = 2
                prf1_startcol = 1
            columns1 = [item.replace('~~~',' ') if item=='~~~' else item  for item in columns1]
            start_row = 2
            sheet_name = '{}'.format(str(n_fold)+'_folds_average' if fold_idx==n_fold else str(fold_idx+1)+'_fold')
            #sheet_name = sheet_name[-20:] if len(sheet_name) >= 31 else sheet_name #Excel worksheet name must be <= 31 chars
            df = pd.DataFrame(data=columns1).transpose() #index=index,
            df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=columns1_2_startcol, index=False, header=False)
            start_row += len(df)
            df = pd.DataFrame(data=columns2).transpose() #index=index,
            df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=columns1_2_startcol, index=False, header=False)
            start_row += len(df)
            for res_idx, res in enumerate(ner_result_dict.keys()):
                if res == 'processed':
                    #ignore 'processed'
                    continue
                for dt_idx, dt in enumerate(test_results_dict.keys()):
                    if res in ['micro_overall', 'macro_overall']:
                        # the first [0]-index item of overall is accuracy, [1:4)-index items will be P/R/F1
                        prf1_indexes = {'start':1,'end':4}
                        rt_idx_exact = 0
                        prf1_exact = ['{:.03f}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx_exact][res_idx][prf1_indexes['start']:prf1_indexes['end']]] #P,R,F1
                        tpfpfn_indexes = {'start':4,'end':7}                        
                        tpfpfn_exact = ['{}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx_exact][res_idx][tpfpfn_indexes['start']:tpfpfn_indexes['end']]] #TP, FP, FN
                        rt_idx_relax = 1
                        prf1_relax = ['{:.03f}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx_relax][res_idx][prf1_indexes['start']:prf1_indexes['end']]] #P,R,F1
                        tpfpfn_relax = ['{}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx_relax][res_idx][tpfpfn_indexes['start']:tpfpfn_indexes['end']]] #TP, FP, FN
                    else:
                        # the first [0:3)-index item of overall is P/R/F1, [3:6)-index items will be right/predict/gold
                        prf1_indexes = {'start':3,'end':6}
                        rt_idx_exact = 0
                        prf1_exact = ['{:.03f}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx_exact][res_idx][prf1_indexes['start']:prf1_indexes['end']]] #P,R,F1
                        tpfpfn_indexes = {'start':4,'end':7}                        
                        tpfpfn_exact = ['{}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx_exact][res_idx][tpfpfn_indexes['start']:tpfpfn_indexes['end']]] #TP, FP, FN
                        rt_idx_relax = 1
                        prf1_relax = ['{:.03f}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx_relax][res_idx][prf1_indexes['start']:prf1_indexes['end']]] #P,R,F1
                    if exact_relax_style == 'horizontal_separate':
                        ner_n_fold_with_average_summarize[res].extend(prf1_exact)
                        ner_n_fold_with_average_summarize[res].extend(tpfpfn_exact)
                        ner_n_fold_with_average_summarize[res].extend(prf1_relax)
                        ner_n_fold_with_average_summarize[res].extend(tpfpfn_relax)
                    elif exact_relax_style == 'vertical':
                        ner_n_fold_with_average_summarize[res+' exact'].extend(prf1_exact)
                        ner_n_fold_with_average_summarize[res+' exact'].extend(tpfpfn_exact)
                        ner_n_fold_with_average_summarize[res+' relax'].extend(prf1_relax)
                        ner_n_fold_with_average_summarize[res+' relax'].extend(tpfpfn_relax)
            #index = ner_tags 
            #index.append('overall') 
            df = pd.DataFrame(data=ner_n_fold_with_average_summarize).transpose() #index=index,
            #df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=index, columns=columns2 )
            df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=prf1_startcol, header=False) #, index=index)
            start_row += len(df) + 1

def write_summarized_results_colleval(summarized_results_xls_pfn, n_fold, ner_tags, test_results_dict, test_results_np):
    #ner_n_fold_with_average_summarize = [OrderedDict()] * (n_fold+1)
    ner_n_fold_with_average_summarize = OrderedDict()
    #n_fold_average_xsl_result_pfn = os.path.join(summarized_result_path, 'ner_i2b2_n2c2_synth_{}_fold_with_average.xlsx'.format(n_fold))
    print('writing ' + summarized_results_xls_pfn)
    with pd.ExcelWriter(summarized_results_xls_pfn) as writer:
        for fold_idx in range(n_fold+1):
            # add ner_tags at first
            for tag in ner_tags:
                ner_n_fold_with_average_summarize[tag] = []        
            # add 'micro/macro overall' at last 
            ner_n_fold_with_average_summarize['micro_overall'] = []
            ner_n_fold_with_average_summarize['macro_overall'] = []
            columns1 = ','.join(["{},~~~,~~~,~~~,~~~,~~~,~~~".format(col) for col in test_results_dict.keys()]).split(',')
            columns1 = [item.replace('~~~',' ') if item=='~~~' else item  for item in columns1]
            columns2 = ['P', 'R', 'F1', 'TP', 'FP', 'FN', 'TN'] * (len(test_results_dict.keys()))
            start_row = 2
            sheet_name = '{}'.format(str(n_fold)+'_folds_average' if fold_idx==n_fold else str(fold_idx+1)+'_fold')
            #sheet_name = sheet_name[-20:] if len(sheet_name) >= 31 else sheet_name #Excel worksheet name must be <= 31 chars
            df = pd.DataFrame(data=columns1).transpose() #index=index,
            df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=1, index=False, header=False)
            start_row += len(df)
            df = pd.DataFrame(data=columns2).transpose() #index=index,
            df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=1, index=False, header=False)            
            start_row += len(df)
            for res_idx, res in enumerate(ner_result_dict.keys()):            
                if res == 'processed':
                    #ignore 'processed'
                    continue
                #ner_n_fold_with_average_summarize[fold_idx][res] = []
                rt_idx = 0 # result_type for exact type
                for dt_idx, dt in enumerate(test_results_dict.keys()):
                    if res == 'accuracy':
                        # the first and only [0]-index item is accuracy
                        accuracy_indexes = {'start':0, 'end': 1}
                        accuracy = ['{:.3f}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx][res_idx][accuracy_indexes['start']:accuracy_indexes['end']]]
                        print(f'accuracy:{accuracy}' )
                        continue                        
                    elif res == 'overall':
                        # the [0:3)-index items are P/R/F1
                        prf1_indexes = {'start':0,'end':3}
                        prf1 = ['{:.3f}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx][res_idx][prf1_indexes['start']:prf1_indexes['end']]] #P,R,F1
                        tpfpfntn_indexes = {'start':3, 'end':7} #TP, FP, FN, TN
                        tpfpfntn = ['{}'.format(int(item)) for item in test_results_np[fold_idx][dt_idx][rt_idx][res_idx][tpfpfntn_indexes['start']:tpfpfntn_indexes['end']]] 
                    else:
                        # the [0:3)-index items are P/R/F1
                        prf1_indexes = {'start':0,'end':3}
                        prf1 = ['{:.3f}'.format(item) for item in test_results_np[fold_idx][dt_idx][rt_idx][res_idx][prf1_indexes['start']:prf1_indexes['end']]] #P,R,F1
                        tpfpfntn_indexes = {'start':3, 'end':7} #TP, FP, FN, TN
                        tpfpfntn = ['{}'.format(int(item)) for item in test_results_np[fold_idx][dt_idx][rt_idx][res_idx][tpfpfntn_indexes['start']:tpfpfntn_indexes['end']]] 
                    prf1_tpfpfntn = prf1
                    prf1_tpfpfntn.extend(tpfpfntn)
                    ner_n_fold_with_average_summarize[res].extend(prf1_tpfpfntn)
            #index = ner_tags 
            #index.append('overall') 
            df = pd.DataFrame(data=ner_n_fold_with_average_summarize).transpose() #index=index,
            #df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=index, columns=columns2 )
            df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, header=False) #, index=index)
            start_row += len(df) + 1
            #overal_accuracy_row = 'overall accuracy: {}'.format(overall_accuracy)
            #df = pd.DataFrame(data=overal_accuracy_row).transpose() #index=index,
            #df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=1, index=False, header=False)            
            #start_row += len(df) + 1


def write_summarized_results(eval_type, summarized_results_xls_pfn, n_fold, ner_tags, test_results_dict, test_results, exact_relax_style='horizontal_separate'):
    if eval_type == 'clamp_tfner_eval':
        write_summarized_results_clamp_dockersimple(summarized_results_xls_pfn, n_fold, ner_tags, test_results_dict, test_results, exact_relax_style)
    elif eval_type == 'clamp_eval':
        write_summarized_results_clamp_eval(summarized_results_xls_pfn, n_fold, ner_tags, test_results_dict, test_results, exact_relax_style)
    else:
        write_summarized_results_colleval(summarized_results_xls_pfn, n_fold, ner_tags, test_results_dict, test_results)

def evaluate_test_predict_results(run_eval_prog, eval_path_prog, eval_type, test_predict_results_pfn, test_score_results_pfn, sep_tag_type='tab'):
    # evaluate preds_voting results
    if eval_type == 'conlleval':
        cmd_str = f'{run_eval_prog} {eval_path_prog} < {test_predict_results_pfn} | tee {test_score_results_pfn}'
        print(cmd_str)
        os.system(cmd_str)
    else:
        evaluate(test_predict_results_pfn, test_score_results_pfn, sep_tag_type )
        #cmd_str = f'{run_eval_prog} {eval_path_prog} -lf {test_predict_results_pfn} -ef {test_score_results_pfn} -s {sep_tag_type}'
        #print(cmd_str)
        #os.system(cmd_str)


if __name__ == "__main__":
    cur_path = os.path.dirname(os.path.realpath(__file__))    
    root_path = os.path.join(cur_path, '..')
    data_root_path = os.path.join(root_path, 'data')
    result_path = os.path.join(root_path, 'result')
    #data_root_path = os.path.join(root_path, 'data')
    #data_root_path = r'D:\Google\DriveUTH\uth\tasks\clamp-cmd-pipeline\bitbucket\test\data_clamp_corpus\drug_entity_1200\xmi'
    tasks = {
        'litcoin_nlp': {
            'data_root_path': r'/raid/jli34/tasks/LitCoinNLP/data/converted/ner/output/ner',
            'result_path': result_path,
            'task_name':  'litcoin_nlp', #ner_ctg
            'eval_type': 'clamp_eval', #colleval, clamp_eval, clamp_tfner_eval
            'eval_path': '', 
            'eval_prog': '',
            'run_eval_prog': 'python', 
            'n_fold': 10,
            'exact_relax_style': 'vertical', # display style for exact and realx results in summarized_results_xls_fn
            'summarized_result_subfold': '{task_name}_result',
            'summarized_results_xls_fn': 'ner_{task_name}_{eval_type}_{n_fold}_fold_with_average_{exact_relax_style}.xlsx',
            'summarized_results_pkl_fn': 'ner_{task_name}_{eval_type}_{n_fold}_fold_with_average_{exact_relax_style}.pkl',                
            'test_result_subdir': 'fold{fold_idx}/output_update_256/{dataset}',
            'test_score_result_fn': 'eval_score_test.txt',
            'test_predict_result_fn': 'predict_test_label',
            'test_pred_sep_tag_type': 'tab',
            'source_voting_result_subdir' : '',
            'source_voting_predict_fn': '',
            'voting_result_subdir': '',
            'voting_all_predict_fn': '',
            'voting_predict_fn': '',
            'voting_result_fn' : '',
            'source_pred_sep_tag_type': 'tab',
            'voting_pred_sep_tag_type' : '',
            'voting_pred_nfold_sep_tag_type': '',
            'source_same_fold_result_subdir': '',
            # datassets list which needs to voting-combine 10folds transfer learning results at first
            'voting_nfold_results_datasets': [
            ],
            'apply_same_fold_results_datasets': [
            ],
            'result_index': None, #depens on eval_type in results_index
            'ner_tags' : ['OrganismTaxon', 'GeneOrGeneProduct', 'CellLine', 'SequenceVariant', 'DiseaseOrPhenotypicFeature', 'ChemicalEntity'],
            #processed_dict = OrderedDict([('tokens',0), ('phrases', 0), ('found', 0), ('correct', 0)])
            'accuracy_dict' : OrderedDict([('accuracy', 0),]),
            'overall_dict' : OrderedDict([('precision', 0), ('recall', 0), ('fb1', 0), ('TP', 0), ('FP', 0), ('FN', 0), ('FN', 0), ('TN', 0),]),
            'ner_tag_dict' : OrderedDict([('precision', 0), ('recall', 0), ('fb1', 0), ('TP', 0), ('FP', 0), ('FN', 0), ('FN', 0), ('TN', 0),]),
            'ner_result_dict' : OrderedDict([('accuracy', 'accuracy_dict'), ('micro_overall', 'overall_dict'), ('macro_overall', 'overall_dict'), \
                                             ('OrganismTaxon', 'ner_tag_dict'), ('GeneOrGeneProduct', 'ner_tag_dict'), ('CellLine', 'ner_tag_dict'), ('SequenceVariant', 'ner_tag_dict'), ('DiseaseOrPhenotypicFeature', 'ner_tag_dict'), ('ChemicalEntity', 'ner_tag_dict')]), #OrderedDict([('micro_overall', copy.deepcopy(overall_dict)), ('macro_overall', copy.deepcopy(overall_dict)), ('drug', copy.deepcopy(ner_tag_dict)), ('problem', copy.deepcopy(ner_tag_dict)), ('test', copy.deepcopy(ner_tag_dict)), ('treatment', copy.deepcopy(ner_tag_dict))]),
            #ner_results_dict = copy.deepcopy(ner_result_dict) # for conlleval results
            'ner_results_dict' : OrderedDict([('exact', 'ner_result_dict'), ('relax', 'ner_result_dict')]),
            'test_results_dict' : OrderedDict([
                ('sultan-BioM-ELECTRA-Large-Discriminator', 'ner_results_dict'), 
                ('sultan-BioM-ALBERT-xxlarge-PMC', 'ner_results_dict'), 
                #copy.deepcopy(ner_results_dict)),\
            ]),
        },

        'clinical_pipeline_data': {
            'data_root_path': r'/home/jli34/data/clamp/submit/dockersimple/data-clinical-pipeline/',
            'result_path': result_path,
            'task_name':  'clamp_dockersimple', #ner_ctg
            'eval_type': 'clamp_tfner_eval', #colleval, clamp_eval
            'eval_path': '', 
            'eval_prog': '',
            'run_eval_prog': 'python', 
            'n_fold': 1,
            'exact_relax_style': 'vertical', # display style for exact and realx results in summarized_results_xls_fn
            'summarized_result_subfold': '{task_name}_result',
            'summarized_results_xls_fn': 'ner_{task_name}_{eval_type}_{n_fold}_fold_with_average_{exact_relax_style}.xlsx',
            'summarized_results_pkl_fn': 'ner_{task_name}_{eval_type}_{n_fold}_fold_with_average_{exact_relax_style}.pkl',                
            'test_result_subdir': '{dataset}',
            'test_score_result_fn': 'test.eval.scores.txt',
            'test_predict_result_fn': 'testb.preds.txt',
            'test_pred_sep_tag_type': 'tab',
            'source_voting_result_subdir' : '',
            'source_voting_predict_fn': '',
            'voting_result_subdir': '',
            'voting_all_predict_fn': '',
            'voting_predict_fn': '',
            'voting_result_fn' : '',
            'source_pred_sep_tag_type': 'tab',
            'voting_pred_sep_tag_type' : '',
            'voting_pred_nfold_sep_tag_type': '',
            'source_same_fold_result_subdir': '',
            # datassets list which needs to voting-combine 10folds transfer learning results at first
            'voting_nfold_results_datasets': [
            ],
            'apply_same_fold_results_datasets': [
            ],
            'result_index': None, #depens on eval_type in results_index
            'ner_tags' : ['drug', 'problem', 'test', 'treatment'],
            #processed_dict = OrderedDict([('tokens',0), ('phrases', 0), ('found', 0), ('correct', 0)])
            'accuracy_dict' : OrderedDict([('accuracy', 0),]),
            'overall_dict' : OrderedDict([('precision', 0), ('recall', 0), ('fb1', 0), ('TP', 0), ('FP', 0), ('FN', 0), ('FN', 0), ('TN', 0),]),
            'ner_tag_dict' : OrderedDict([('precision', 0), ('recall', 0), ('fb1', 0), ('TP', 0), ('FP', 0), ('FN', 0), ('FN', 0), ('TN', 0),]),
            'ner_result_dict' : OrderedDict([('accuracy', 'accuracy_dict'), ('micro_overall', 'overall_dict'), ('macro_overall', 'overall_dict'), ('drug', 'ner_tag_dict'), ('problem', 'ner_tag_dict'), ('test', 'ner_tag_dict'), ('treatment', 'ner_tag_dict')]), #OrderedDict([('micro_overall', copy.deepcopy(overall_dict)), ('macro_overall', copy.deepcopy(overall_dict)), ('drug', copy.deepcopy(ner_tag_dict)), ('problem', copy.deepcopy(ner_tag_dict)), ('test', copy.deepcopy(ner_tag_dict)), ('treatment', copy.deepcopy(ner_tag_dict))]),
            #ner_results_dict = copy.deepcopy(ner_result_dict) # for conlleval results
            'ner_results_dict' : OrderedDict([('exact', 'ner_result_dict'), ('relax', 'ner_result_dict')]),
            'test_results_dict' : OrderedDict([
                ('char_lstm_glove6B_ep300_dr0.5', 'ner_results_dict'), #copy.deepcopy(ner_results_dict)),\
            ]),
        },

        'drug_entity_1200_xmi': {             
            'data_root_path': r'/data/jli34/clamp/dockersimple/data_clamp_corpus/drug_entity_1200_xmi/xmi2/xmi',
            'result_path': result_path,            
            'task_name':  'clamp_dockersimple', #ner_ctg
            'eval_type': 'clamp_tfner_eval', #colleval, clamp_eval
            'eval_path': '', 
            'eval_prog': '',
            'run_eval_prog': 'python', 
            'n_fold': 10,
            'exact_relax_style': 'vertical', # display style for exact and realx results in summarized_results_xls_fn
            'summarized_result_subfold': '{task_name}_result',
            'summarized_results_xls_fn': 'ner_{task_name}_{eval_type}_{n_fold}_fold_with_average_{exact_relax_style}.xlsx',
            'summarized_results_pkl_fn': 'ner_{task_name}_{eval_type}_{n_fold}_fold_with_average_{exact_relax_style}.pkl',                
            'test_result_subdir': 'fold_{fold_idx}/models/{dataset}_fd{fold_idx}',                        
            'test_score_result_fn': 'test.eval.scores.txt',
            'test_predict_result_fn': 'testb.preds.txt',
            'test_pred_sep_tag_type': 'tab',
            'source_voting_result_subdir' : 'output/epoch25/source_fold_{idx}/score',
            'source_voting_predict_fn': 'testb.preds.txt',
            'voting_result_subdir': 'output/epoch25/score_voting',
            'voting_all_predict_fn': 'testb.all.preds.txt',
            'voting_predict_fn': 'testb.preds.txt',
            'voting_result_fn' : 'score.testb.metrics.txt',            
            'source_pred_sep_tag_type': 'tab',
            'voting_pred_sep_tag_type' : 'tab',
            'voting_pred_nfold_sep_tag_type': 'space',
            'source_same_fold_result_subdir': 'output/epoch25/source_fold_{idx}/score',
            # datassets list which needs to voting-combine 10folds transfer learning results at first
            'voting_nfold_results_datasets': [
                'i2b2_hpi_for_utnotes_hpi',
                'i2b2_hpi_for_mtsamples_hpi',
                'i2b2_hpi_for_i2b2_n2c2_synth',           
                'i2b2_n2c2_synth_for_utnotes_hpi',
                'i2b2_n2c2_synth_for_mtsamples_hpi',
                'i2b2_n2c2_synth_for_i2b2_hpi',         
            ],
            'apply_same_fold_results_datasets': [
                'i2b2_hpi_synth_combined_for_utnotes_hpi',
                'i2b2_hpi_synth_combined_for_mtsamples_hpi',
                'i2b2_hpi_synth_combined_for_i2b2_hpi', 
                'i2b2_hpi_synth_combined_for_i2b2_n2c2_synth',         
            ],
            'result_index' : None,
            'ner_tags' : ['drug', 'problem', 'test', 'treatment'],
            #processed_dict = OrderedDict([('tokens',0), ('phrases', 0), ('found', 0), ('correct', 0)])
            'accuracy_dict' : OrderedDict([('accuracy', 0),]),
            'overall_dict' : OrderedDict([('precision', 0), ('recall', 0), ('fb1', 0), ('TP', 0), ('FP', 0), ('FN', 0), ('FN', 0), ('TN', 0),]),
            'ner_tag_dict' : OrderedDict([('precision', 0), ('recall', 0), ('fb1', 0), ('TP', 0), ('FP', 0), ('FN', 0), ('FN', 0), ('TN', 0),]),
            'ner_result_dict' : OrderedDict([('accuracy', 'accuracy_dict'), ('micro_overall', 'overall_dict'), ('macro_overall', 'overall_dict'), ('drug', 'ner_tag_dict'), ('problem', 'ner_tag_dict'), ('test', 'ner_tag_dict'), ('treatment', 'ner_tag_dict')]), #OrderedDict([('micro_overall', copy.deepcopy(overall_dict)), ('macro_overall', copy.deepcopy(overall_dict)), ('drug', copy.deepcopy(ner_tag_dict)), ('problem', copy.deepcopy(ner_tag_dict)), ('test', copy.deepcopy(ner_tag_dict)), ('treatment', copy.deepcopy(ner_tag_dict))]),
            #ner_results_dict = copy.deepcopy(ner_result_dict) # for conlleval results
            'ner_results_dict' : OrderedDict([('exact', 'ner_result_dict'), ('relax', 'ner_result_dict)')]),
            'test_results_dict' : OrderedDict([
                ('char_lstm_pubmed_ep100_dr0.3', 'ner_results_dict'), #copy.deepcopy(ner_results_dict)),\
                ('char_lstm_pubmed_ep100_dr0.5', 'ner_results_dict'), #copy.deepcopy(ner_results_dict)), \
                ('char_lstm_pubmed_ep100_dr0.8', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_pubmed_ep200_dr0.3', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_pubmed_ep200_dr0.5', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_pubmed_ep200_dr0.8', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_pubmed_ep300_dr0.3', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_pubmed_ep300_dr0.5', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_pubmed_ep300_dr0.8', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_w2v_ep100_dr0.3', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_w2v_ep100_dr0.5', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_w2v_ep100_dr0.8', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_w2v_ep200_dr0.3', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_w2v_ep200_dr0.5', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_w2v_ep200_dr0.8', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_w2v_ep300_dr0.3', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_w2v_ep300_dr0.5', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
                ('char_lstm_w2v_ep300_dr0.8', 'ner_results_dict'), # copy.deepcopy(ner_results_dict)), \
            ]),
        },
        'ctg_ner_colleval': {
            'data_root_path': data_root_path,
            'result_path': result_path,            
            'task_name':  'ctg_ner', #clamp_dockersimple
            'eval_type': 'colleval', #clamp_tfner_eval, clamp_eval
            'eval_path': os.path.join(root_path, 'chars_bilstm_crf'),
            'eval_prog' : 'conlleval', 
            'run_eval_prog' : 'perl', 
            
            'n_fold': 10,
            'exact_relax_style': 'vertical', # display style for exact and realx results in summarized_results_xls_fn
            'summarized_result_subfold': '{task_name}_result',
            'summarized_results_xls_fn': 'ner_{task_name}_{eval_type}_{n_fold}_fold_with_average_{exact_relax_style}.xlsx',
            'summarized_results_pkl_fn': 'ner_{task_name}_{eval_type}_{n_fold}_fold_with_average_{exact_relax_style}.pkl',                
            'test_result_subdir': 'output/epoch25/score',             
            'test_score_result_fn': 'score.testb.metrics.txt',
            'test_predict_result_fn': 'testb.preds.txt',
            'test_pred_sep_tag_type': 'space',
            'source_voting_result_subdir' : 'output/epoch25/source_fold_{idx}/score',
            'source_voting_predict_fn': 'testb.preds.txt',
            'voting_result_subdir': 'output/epoch25/score_voting',
            'voting_all_predict_fn': 'testb.all.preds.txt',
            'voting_predict_fn': 'testb.preds.txt',
            'voting_result_fn' : 'score.testb.metrics.txt',
            'source_pred_sep_tag_type': 'space',
            'voting_pred_sep_tag_type': 'tab',
            'voting_pred_nfold_sep_tag_type': 'space',
            'source_same_fold_result_subdir': 'output/epoch25/source_fold_{fold_idx}/score',
            # datassets list which needs to voting-combine 10folds transfer learning results at first
            'voting_nfold_results_datasets' : [
                'i2b2_hpi_for_utnotes_hpi',
                'i2b2_hpi_for_mtsamples_hpi',
                'i2b2_hpi_for_i2b2_n2c2_synth',           
                'i2b2_n2c2_synth_for_utnotes_hpi',
                'i2b2_n2c2_synth_for_mtsamples_hpi',
                'i2b2_n2c2_synth_for_i2b2_hpi',         
            ],
            'apply_same_fold_results_datasets' : [
                'i2b2_hpi_synth_combined_for_utnotes_hpi',
                'i2b2_hpi_synth_combined_for_mtsamples_hpi',
                'i2b2_hpi_synth_combined_for_i2b2_hpi', 
                'i2b2_hpi_synth_combined_for_i2b2_n2c2_synth',         
            ],
            'result_index': None, #depens on eval_type in results_index 
            'ner_tags' : ['problem', 'test', 'treatment'],
            #processed_dict = OrderedDict([('tokens',0), ('phrases', 0), ('found', 0), ('correct', 0)])
            'accuracy_dict' : OrderedDict([('accuracy', 0),]),
            'overall_dict' : OrderedDict([('precision', 0), ('recall', 0), ('fb1', 0), ('TP', 0), ('FP', 0), ('FN', 0), ('TN', 0),]),
            'ner_tag_dict' : OrderedDict([('precision', 0), ('recall', 0), ('fb1', 0), ('TP', 0), ('FP', 0), ('FN', 0), ('TN', 0),]),
            'ner_result_dict' : OrderedDict([('accuracy', 'accuracy_dict'), ('micro_overall', 'overall_dict'), ('macro_overall', 'overall_dict'), ('problem', 'ner_tag_dict'), ('test', 'ner_tag_dict'), ('treatment', 'ner_tag_dict')]), #OrderedDict([('micro_overall', copy.deepcopy(overall_dict)), ('macro_overall', copy.deepcopy(overall_dict)), ('drug', copy.deepcopy(ner_tag_dict)), ('problem', copy.deepcopy(ner_tag_dict)), ('test', copy.deepcopy(ner_tag_dict)), ('treatment', copy.deepcopy(ner_tag_dict))]),
            #ner_results_dict = copy.deepcopy(ner_result_dict) # for conlleval results
            'ner_results_dict': OrderedDict([('exact', 'ner_result_dict'), ('relax', 'ner_result_dict')]), #OrderedDict([('exact', 'ner_result_dict'), ('relax', 'ner_result_dict)')]),
            'test_results_dict' : OrderedDict([
                ('i2b2_n2c2_synth', 'ner_results_dict'), #copy.deepcopy(ner_result_dict)), \
                ('i2b2_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('utnotes_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('mtsamples_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('i2b2_hpi_synth_combined', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('i2b2_hpi_for_utnotes_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('i2b2_hpi_for_mtsamples_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('i2b2_hpi_for_i2b2_n2c2_synth', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('i2b2_n2c2_synth_for_utnotes_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('i2b2_n2c2_synth_for_mtsamples_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('i2b2_n2c2_synth_for_i2b2_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('i2b2_hpi_synth_combined_for_utnotes_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('i2b2_hpi_synth_combined_for_mtsamples_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('i2b2_hpi_synth_combined_for_i2b2_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
                ('i2b2_hpi_synth_combined_for_i2b2_n2c2_synth', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)), \
            ]),
        },
        'ctg_ner_clamp_eval': {
            'data_root_path': data_root_path,
            'result_path': result_path,
            'task_name':  'ctg_ner', #clamp_dockersimple
            'eval_type': 'clamp_eval', #clamp_tfner_eval, colleval
            'eval_path': cur_path,
            'eval_prog' : 'evaluation_new.py', 
            'run_eval_prog' : 'python', 
            
            'n_fold': 10,
            'exact_relax_style': 'horizontal_combine', # display style for exact and realx results in summarized_results_xls_fn
            'summarized_result_subfold': '{task_name}_result',
            'summarized_results_xls_fn': 'ner_{task_name}_{eval_type}_{n_fold}_fold_with_average_{exact_relax_style}.xlsx',
            'summarized_results_pkl_fn': 'ner_{task_name}_{eval_type}_{n_fold}_fold_with_average_{exact_relax_style}.pkl',                
            'test_result_subdir': 'output/epoch25/score',             
            'test_score_result_fn': 'score.testb.metrics.clamp_eval.txt',
            'test_predict_result_fn': 'testb.preds.txt',
            'test_pred_sep_tag_type': 'space',
            'source_voting_result_subdir' : 'output/epoch25/source_fold_{idx}/score',
            'source_voting_predict_fn': 'testb.preds.txt',
            'voting_result_subdir': 'output/epoch25/score_voting',
            'voting_all_predict_fn': 'testb.all.preds.txt',
            'voting_predict_fn': 'testb.preds.txt',
            'voting_result_fn' : 'score.testb.metrics.clamp_eval.txt',
            'source_pred_sep_tag_type': 'space',
            'voting_pred_sep_tag_type': 'tab',
            'voting_pred_nfold_sep_tag_type': 'space',
            'source_same_fold_result_subdir': 'output/epoch25/source_fold_{fold_idx}/score',
            # datassets list which needs to voting-combine 10folds transfer learning results at first
            'voting_nfold_results_datasets' : [
                'i2b2_hpi_for_utnotes_hpi',
                'i2b2_hpi_for_mtsamples_hpi',
                'i2b2_hpi_for_i2b2_n2c2_synth',           
                'i2b2_n2c2_synth_for_utnotes_hpi',
                'i2b2_n2c2_synth_for_mtsamples_hpi',
                'i2b2_n2c2_synth_for_i2b2_hpi',         
            ],
            'apply_same_fold_results_datasets' : [
                'i2b2_hpi_synth_combined_for_utnotes_hpi',
                'i2b2_hpi_synth_combined_for_mtsamples_hpi',
                'i2b2_hpi_synth_combined_for_i2b2_hpi', 
                'i2b2_hpi_synth_combined_for_i2b2_n2c2_synth',         
            ],
            'result_index': None, #depens on eval_type in results_index
            'ner_tags' : ['problem', 'test', 'treatment'],
            #processed_dict = OrderedDict([('tokens',0), ('phrases', 0), ('found', 0), ('correct', 0)])
            'accuracy_dict' : OrderedDict([('accuracy', 0),]),
            'overall_dict' : OrderedDict([('precision', 0), ('recall', 0), ('fb1', 0), ('TP', 0), ('FP', 0), ('FN', 0), ('TN', 0),]),
            'ner_tag_dict' : OrderedDict([('precision', 0), ('recall', 0), ('fb1', 0), ('TP', 0), ('FP', 0), ('FN', 0), ('TN', 0),]),
            'ner_result_dict' : OrderedDict([('accuracy', 'accuracy_dict'), ('micro_overall', 'overall_dict'), ('macro_overall', 'overall_dict'), ('problem', 'ner_tag_dict'), ('test', 'ner_tag_dict'), ('treatment', 'ner_tag_dict')]), #OrderedDict([('micro_overall', copy.deepcopy(overall_dict)), ('macro_overall', copy.deepcopy(overall_dict)), ('drug', copy.deepcopy(ner_tag_dict)), ('problem', copy.deepcopy(ner_tag_dict)), ('test', copy.deepcopy(ner_tag_dict)), ('treatment', copy.deepcopy(ner_tag_dict))]),
            #ner_results_dict = copy.deepcopy(ner_result_dict) # for conlleval results
            'ner_results_dict' : OrderedDict([('exact', 'ner_result_dict'), ('relax', 'ner_result_dict')]),
            'test_results_dict' : OrderedDict([
                ('i2b2_n2c2_synth', 'ner_results_dict'), #copy.deepcopy(ner_result_dict)),
                ('i2b2_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('utnotes_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('mtsamples_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_hpi_synth_combined', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_hpi_for_utnotes_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_hpi_for_mtsamples_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_hpi_for_i2b2_n2c2_synth', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_n2c2_synth_for_utnotes_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_n2c2_synth_for_mtsamples_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_n2c2_synth_for_i2b2_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_hpi_synth_combined_for_utnotes_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_hpi_synth_combined_for_mtsamples_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_hpi_synth_combined_for_i2b2_hpi', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_hpi_synth_combined_for_i2b2_n2c2_synth', 'ner_results_dict'), # copy.deepcopy(ner_result_dict)),
                ('i2b2_hpi_synth_for_i2b2_jamia_1', 'ner_results_dict'),
            ]),
        },

    }
    test_tasks = ['litcoin_nlp',]
    test_results = []
    for tt in test_tasks:
        task = tasks[tt]
        data_root_path = task['data_root_path']
        result_path = task['result_path']
        task_name = task['task_name']
        eval_type = task['eval_type']
        eval_path = task['eval_path']
        eval_prog = task['eval_prog']
        run_eval_prog = task['run_eval_prog']
        eval_path_prog = os.path.join(eval_path, eval_prog)
        n_fold = task['n_fold']
        exact_relax_style = task['exact_relax_style']
        summarized_result_subfold = task['summarized_result_subfold'].format(task_name=task_name)
        summarized_result_path = os.path.join(result_path, summarized_result_subfold)
        os.makedirs(summarized_result_path, exist_ok=True)
        summarized_results_xls_fn = task['summarized_results_xls_fn'].format(task_name=task_name, eval_type=eval_type, n_fold=n_fold, exact_relax_style=exact_relax_style)
        summarized_results_xls_pfn = os.path.join(summarized_result_path, summarized_results_xls_fn)
        summarized_results_pkl_fn = task['summarized_results_pkl_fn'].format(task_name=task_name, eval_type=eval_type, n_fold=n_fold, exact_relax_style=exact_relax_style)
        summarized_results_pkl_pfn = os.path.join(summarized_result_path, summarized_results_pkl_fn)
        
        test_result_subdir = task['test_result_subdir'] #'fold_{fold_idx}/models/{dataset}_fd{fold_idx}'
        test_score_result_fn = task['test_score_result_fn'] #'test.eval.scores.txt'
        test_predict_result_fn = task['test_predict_result_fn'] #'testb.preds.txt'
        test_pred_sep_tag_type = task['test_pred_sep_tag_type'] 
        source_voting_result_subdir = task['source_voting_result_subdir']
        source_voting_predict_fn = task['source_voting_predict_fn']
        voting_result_subdir = task['voting_result_subdir']
        voting_all_predict_fn = task['voting_all_predict_fn']
        voting_predict_fn = task['voting_predict_fn']
        voting_result_fn = task['voting_result_fn']        
        source_pred_sep_tag_type = task['source_pred_sep_tag_type']
        voting_pred_sep_tag_type = task['voting_pred_sep_tag_type']
        voting_pred_nfold_sep_tag_type = task['voting_pred_nfold_sep_tag_type']
        source_same_fold_result_subdir = task['source_same_fold_result_subdir']
        # datassets list which needs to voting-combine 10folds transfer learning results at first
        voting_nfold_results_datasets = task['voting_nfold_results_datasets']
        apply_same_fold_results_datasets = task['apply_same_fold_results_datasets']
        result_index = results_index[eval_type]
        task['result_index'] = result_index
        ner_tags = task['ner_tags']
        #processed_dict = OrderedDict([('tokens',0), ('phrases', 0), ('found', 0), ('correct', 0)])
        overall_dict = task['overall_dict'] 
        ner_tag_dict = task['ner_tag_dict'] 
        ner_result_dict_tmp = task['ner_result_dict']
        ner_result_dict = OrderedDict()
        for ner_res in ner_result_dict_tmp:            
            ner_result_dict[ner_res] = copy.deepcopy(task[ner_result_dict_tmp[ner_res]])
        task['ner_result_dict'] = ner_result_dict
        #ner_results_dict = copy.deepcopy(ner_result_dict) # for conlleval results
        ner_results_dict_tmp = task['ner_results_dict']
        ner_results_dict = OrderedDict()
        for ner_res in ner_results_dict_tmp:
            ner_results_dict[ner_res] = copy.deepcopy(task[ner_results_dict_tmp[ner_res]])
        task['ner_results_dict'] = ner_results_dict
        test_results_dict_tmp = task['test_results_dict']
        test_results_dict = OrderedDict()
        for test_res in test_results_dict_tmp:
            test_results_dict[test_res] = copy.deepcopy(task[test_results_dict_tmp[test_res]])
        task['test_results_dict'] = test_results_dict
        for fold_idx in range(n_fold):
            for dt in test_results_dict.keys():
                print(f'{tt}:{fold_idx}:{dt}')
                #data_root_path = os.path.join(data_root_path, dt, 'fold_{}'.format(fold_idx+1))
                if dt in voting_nfold_results_datasets:
                    vote_combining_nfold_results(
                            n_fold, data_root_path, 
                            source_voting_result_subdir, 
                            source_voting_predict_fn, 
                            voting_result_subdir, 
                            voting_all_predict_fn, 
                            voting_predict_fn, 
                            voting_result_fn,
                            run_eval_prog, 
                            eval_path_prog, 
                            source_pred_sep_tag_type, 
                            voting_pred_sep_tag_type, 
                            voting_pred_nfold_sep_tag_type)
                    test_score_result_pfn = os.path.join(data_root_path, voting_result_subdir, voting_result_fn)
                    test_predict_result_pfn = os.path.join(data_root_path, voting_result_subdir, voting_predict_fn) 
                elif dt in apply_same_fold_results_datasets:
                    test_score_result_pfn = os.path.join(data_root_path, source_same_fold_result_subdir.format(fold_idx=fold_idx+1), voting_result_fn)
                    test_predict_result_pfn = os.path.join(data_root_path, source_same_fold_result_subdir.format(fold_idx=fold_idx+1), voting_predict_fn)
                else:
                    test_score_result_pfn = os.path.join(data_root_path, test_result_subdir.format(fold_idx=fold_idx+1, dataset=dt), test_score_result_fn)
                    test_predict_result_pfn = os.path.join(data_root_path, test_result_subdir.format(fold_idx=fold_idx+1, dataset=dt), test_predict_result_fn)
                print(test_score_result_pfn)
                if not os.path.exists(test_score_result_pfn) and os.path.exists(test_predict_result_pfn):
                    # re-calculate eval-score results
                    evaluate_test_predict_results(run_eval_prog, eval_path_prog, eval_type, test_predict_result_pfn, test_score_result_pfn, test_pred_sep_tag_type)
                if os.path.exists(test_score_result_pfn):
                    with open(test_score_result_pfn, 'r') as rf:
                        results = rf.read().split('\n')
                        print('{}:{}'.format(dt, results))
                else:
                    results = []
                    
                test_result = get_test_results(results, task)
                test_results_dict[dt] = copy.deepcopy(test_result)
            test_results.append(copy.deepcopy(test_results_dict))
        test_results_np = np.zeros([n_fold+1, len(test_results_dict), len(ner_results_dict), len(ner_result_dict), max(len(overall_dict), len(ner_tag_dict))], dtype=float)
        #dataset_indexes = OrderedDict([(i2b2_n2c2_synth_dataset, 0), (i2b2_hpi_dataset, 1), (mtsamples_hpi_dataset, 2), \
        #                   (i2b2_n2c2_synth_for_i2b2_hpi_dataset, 3), (i2b2_n2c2_synth_for_mtsamples_hpi_dataset, 4) ])
        for fold in range(n_fold):
            for dt_idx, dt in enumerate(test_results_dict.keys()):
                for rt_idx, rt in enumerate(ner_results_dict.keys()):
                    for res_idx, res in enumerate(ner_result_dict.keys()):
                        #print(test_results[fold][dt][rt][res].values())
                        res_len = len(test_results[fold][dt][rt][res].values())                    
                        test_results_np[fold][dt_idx][rt_idx][res_idx][:res_len] = list(test_results[fold][dt][rt][res].values())
                        test_results_np[n_fold][dt_idx][rt_idx][res_idx] += test_results_np[fold][dt_idx][rt_idx][res_idx]
                        print(test_results_np[fold][dt_idx][rt_idx][res_idx])
                        #print(test_results_np[n_fold][dt_idx][rt_idx][res_idx])

        test_results_np[n_fold] /= n_fold
        #summarized_results_xls_fn = 'ner_i2b2_semeval_tl_10folds.xlsx'
        write_summarized_results(eval_type, summarized_results_xls_pfn, n_fold, ner_tags, test_results_dict, test_results_np, exact_relax_style=exact_relax_style)
        with open(summarized_results_pkl_pfn, 'wb') as wf:
            pickle.dump(test_results_np, wf)
        score_sample = """
            processed 5153 tokens with 587 phrases; found: 557 phrases; correct: 471.
            accuracy:  93.52%; precision:  84.56%; recall:  80.24%; FB1:  82.34
                    problem: precision:  83.28%; recall:  83.03%; FB1:  83.16  329
                        test: precision:  88.35%; recall:  75.21%; FB1:  81.25  103
                    treatment: precision:  84.80%; recall:  77.94%; FB1:  81.23  125
        """
        # write n-fold average in txt file like each fold result
        for dt_idx, dt in enumerate(test_results_dict.keys()):        
            n_fold_average_result_pfn = os.path.join(summarized_result_path, 'ner_{}_{}_fold_average.txt'.format(dt, n_fold))
            with open(n_fold_average_result_pfn, 'w') as awf:
                for rt_idx, rt in enumerate(ner_results_dict.keys()):
                    awf.writelines(rt+'\n')
                    for res_idx, res in enumerate(ner_result_dict.keys()):                
                        if res == 'processed':
                            processed_line = 'processed {:.0f} tokens with {:.0f} phrases; found: {:.0f} phrases; correct: {:.0f}.\n'.format(*test_results_np[n_fold][dt_idx][rt_idx][res_idx])
                            awf.writelines(processed_line)
                        elif res in ['micro_overall', 'macro_overall']:
                            overall_line = '{}: accuracy: {:.3f}%; precision: {:.3f}%; recall: {:.3f}%; FB1: {:.3f}\n'.format(res, *test_results_np[n_fold][dt_idx][rt_idx][res_idx])
                            awf.writelines(overall_line)
                        else:
                            ner_tags_line = '{}: right: {}; predict: {}; gold: {}; precision: {:.2f}%; recall: {:.2f}%; FB1: {:.2f} \n'.format(res, *test_results_np[n_fold][dt_idx][rt_idx][res_idx]) 
                            awf.writelines(ner_tags_line)
    print('combine_train_test_bio_files done.')