import logging
import sys
import os
from unittest.mock import patch
import time

# conda activate litcoin
import torch
from transformers import AutoTokenizer, AutoModel


from run_ner import main
from utils_ner import get_labels, update_data_to_max_len


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

def test_run_ner(paras):
    gpu = paras['gpu']
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)

    testargs = f"""
        --model_type {paras['model_type']}
        --model_name {paras['model_name_or_path']}
        --output_dir {paras['output_dir']}
        --data_dir {paras['data_dir']}
        --train_file {paras['train_file']}
        --dev_file {paras['dev_file']}
        --test_file {paras['test_file']}
        --labels {paras['labels']}
        --max_seq_length {paras['max_seq_len']}
        --num_train_epochs {paras['num_train_epochs']}
        --save_steps {paras['save_steps']}
        --logging_steps {paras['logging_steps']}
    """
    if not paras['cased']:
        testargs += " --do_lower_case"
    if paras['do_train']:
        testargs += " --do_train "
    if paras['do_eval']:            
        testargs += " --do_eval "
    if paras['do_predict']:
        testargs += " --do_predict \n"
    if paras['overwrite_output_dir']:
        testargs += " --overwrite_output_dir "
    if paras['overwrite_cache']:
        testargs += " --overwrite_cache "
    testargs = testargs.split()
    with patch.object(sys, "argv", ["run.py"] + testargs):
        result = main()
        print(result)

if __name__ == "__main__":
    #cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_type_names = {
        'electra': {
            'biom-electra-large': {
                'model_name': 'sultan/BioM-ELECTRA-Large-Discriminator',
                'cased': False,
            },
        },
        'albert': {
            'biom-albert-xxlarge': {
                'model_name': 'sultan/BioM-ALBERT-xxlarge-PMC',
                'cased': False,
            },
        },
        'bert': {
            # english
            'BiomedNLP-PubMedBERT-base-uncased-abstract': {
                'model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', 
                'cased': False,
            },
            'BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext': {
                'model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 
                'cased': False,
            },
            'biobert-large-cased-v1.1': {
                'model_name': 'dmis-lab/biobert-large-cased-v1.1', 
                'cased': True,
            },
            'biobert-base-cased-v1.1': {
                'model_name': 'dmis-lab/biobert-base-cased-v1.1', 
                'cased': True,
            },
            'biobert-base-cased-v1.2': {
                'model_name': 'dmis-lab/biobert-base-cased-v1.2', 
                'cased': True,
            },
            'bert-large-cased': {
                'model_name': 'bert-large-cased', 
                'cased': True,
            },
            'bert-large-uncased': {
                'model_name': 'bert-large-cased', 
                'cased': False,
            },
            'ClinicalBERT': {
                'model_name': 'emilyalsentzer/Bio_ClinicalBERT', 
                'cased': True,
            },
            'bluebert_pubmed_mimic_uncased_L': {
                'model_name': 'bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16', 
                'cased': False,
            },
        },
    }
    tasks = {
        'litcoin_ner': {
            'gpu': '7',
            'nfold': 6,
            'root_dir': '/home/yhu5/jianfu_NER/ner_ft_bert/',
            'eval_prog_fmt': '/home/yhu5/jianfu_NER/ner_ft_bert/evaluate.py',
            'data_root_dir_fmt': '{root_dir}/data/EBM-NLPmod/',
            'datasets': ['5-fold'],
            # labels file name, will be used in run_ner.py
            'labels_fmt': '{data_root_dir}/labels.txt', 
            'data_dir_fmt': '{data_root_dir}/fold{fold}',
            # {update_max_len_suffix} will be determined by do_update_max_len and max_seq_len
            # like _128 or _update_128
            'output_dir_fmt': '{data_root_dir}/output_major_revision_check/fold{fold}/output{update_max_len_suffix}/{model_name}/',
            'model_name_or_path_fmt': '',
            'train_file': 'train.txt',
            'test_file': 'test.txt',
            'dev_file': 'dev.txt',
            'predict_test_file': 'predict_test_label.txt',
            'eval_score_test_file': 'eval_score_test.txt',
            'model_name_or_path': '',            
            'model_type': None,
            'cased': None,
            'test_model_types': {
                'electra': [
                    #'biom-electra-large',
                ],
                'albert': [
                    #'biom-albert-xxlarge',
                ],
                'bert': [
                    #'bert-large-cased',
                    #'bert-large-uncased',
                    'BiomedNLP-PubMedBERT-base-uncased-abstract',
                    #'biobert-large-cased-v1.1',  
                    #'biobert-base-cased-v1.1',
                    #'biobert-base-cased-v1.2',  
                    #'ClinicalBERT',
                    #'bluebert_pubmed_mimic_uncased_L'         
                ]
            },
            'overwrite_output_dir': True,
            'num_train_epochs': 10,
            'save_steps': 20000,
            'logging_steps': 1000,
            'do_train': True,
            'do_eval': True,
            'do_predict': True,
            'overwrite_cache': False,
            'max_seq_len_list': [256],
            'do_update_max_len_list': [True],
            'overwrite_update_max_len_file': True,
            'sep_tag_type' : 'tab', # 'space'
        },  
    }
    
    test_tasks = ['litcoin_ner' ]
    for tt in test_tasks:
        task = tasks[tt]
        nfold = task['nfold']
        #ext = task['ext']
        root_dir = task['root_dir']
        #'{root_dir}/evaluate_new.py',
        eval_prog =  task['eval_prog_fmt'].format(root_dir=root_dir)
        task['eval_prog'] = eval_prog
        datasets = task['datasets']
        for dataset in datasets:
            #'{root_dir}/data/korean-ner/clinical_notes_500_jy/bio/combined/5fold',
            data_root_dir = task['data_root_dir_fmt'].format(root_dir=root_dir, dataset=dataset)
            #'{data_root_dir}/labels.txt', 
            labels_file = task['labels_fmt'].format(data_root_dir=data_root_dir)        
            task['labels'] = labels_file
            for fold_idx in range(5):            
                #'data_dir_fmt': '{data_root_dir}/fold_{fold}/input'
                data_dir = task['data_dir_fmt'].format(data_root_dir=data_root_dir, dataset=dataset, fold=fold_idx+1) 
                print(data_dir)           
                task['data_dir'] = data_dir
                #'output_dir_fmt': '{data_root_dir}/fold_{fold}/output',
                #'{output_dir}/predict_test_label.txt',
                predict_test_file = task['predict_test_file']            
                #'{output_dir}/eval_score_test.txt',
                eval_score_test_file = task['eval_score_test_file']            
                #'{root_dir}/evalaute_new.py',
                sep_tag_type = task['sep_tag_type']
                #file_list = task['file_list']
                train_file = task['train_file']
                dev_file = task['dev_file']
                test_file = task['test_file']
                file_list = [train_file, dev_file, test_file]
                if not os.path.exists(labels_file):
                    labels = get_labels(labels_file, data_dir, file_list)
                    with open(labels_file, 'w', encoding='utf-8') as wf:
                        wf.write('\n'.join(labels))        
                test_model_types = task['test_model_types']            
                for model_type in test_model_types:
                    for test_mn in test_model_types[model_type]:
                        model_name = model_type_names[model_type][test_mn]['model_name']
                        cased = model_type_names[model_type][test_mn]['cased']
                        task['cased'] = cased
                        max_seq_len_list = task['max_seq_len_list']
                        do_update_max_len_list = task['do_update_max_len_list']
                        for max_seq_len in max_seq_len_list:
                            task['max_seq_len'] = max_seq_len
                            for do_update_max_len in do_update_max_len_list:                        
                                task['do_update_max_len'] = do_update_max_len                    
                                if do_update_max_len:
                                    update_max_len_suffix = '_update_{}'.format(max_seq_len)
                                else:
                                    update_max_len_suffix = '_{}'.format(max_seq_len)
                                output_dir = task['output_dir_fmt'].format(data_root_dir=data_root_dir, model_name=model_name.replace('/','-'), update_max_len_suffix=update_max_len_suffix, fold=fold_idx+1)
                                os.makedirs(output_dir, exist_ok=True) 
                                print(output_dir)           
                                task['output_dir'] = output_dir
                                #{root_dir}/models/{model_name}',
                                #model_name_or_path = task['model_name_or_path_fmt'].format(root_dir=root_dir, model_name=model_name)
                                if os.path.exists(task['model_name_or_path']):
                                    task['model_name_or_path'] = task['model_name_or_path']
                                else:
                                    task['model_name_or_path'] = model_name
                                task['model_type'] = model_type
                                predict_test_pfn = os.path.join(task['output_dir'], predict_test_file)
                                eval_score_test_pfn = os.path.join(task['output_dir'], eval_score_test_file)                    
                                start = time.process_time()
                                if not os.path.exists(predict_test_pfn):
                                    # udpate input data to max_seq_len                    
                                    if do_update_max_len:
                                        # truncate input dataset to max_seq_len for each sentence                            
                                        for fn in file_list:
                                            file_pfn = os.path.join(data_dir, fn)
                                            #file_name, file_ext = os.path.splitext(fn)
                                            #file_pfn_new = os.path.join(output_dir, file_name+'_update_{}{}'.format(max_seq_len, file_ext))
                                            overwrite_update_max_len_file = task['overwrite_update_max_len_file']
                                            file_pfn_new = os.path.join(output_dir, fn)
                                            if (not os.path.exists(file_pfn_new)) or \
                                                (os.path.exists(file_pfn_new) and overwrite_update_max_len_file):
                                                update_data_to_max_len(max_seq_len, file_pfn, file_pfn_new, task['model_name_or_path'])
                                        task['data_dir'] = output_dir                                
                                    test_run_ner(task)                                                                    
                                if eval_score_test_file and not os.path.exists(eval_score_test_pfn):
                                    # evaluate using evalutate_new.py
                                    cmd_str = f"python {eval_prog} \
                                        -lf {predict_test_pfn} \
                                        -ef {eval_score_test_pfn} \
                                        -st {sep_tag_type}"
                                    print(cmd_str)
                                    os.system(cmd_str)                                
                                elapsed = time.process_time() - start
                                torch.cuda.empty_cache()
                                print(f'{tt} fold_{fold_idx+1} {model_name} {max_seq_len} update_max_len:{do_update_max_len} done, {elapsed} seconds elasped...')

