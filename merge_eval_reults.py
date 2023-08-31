from glob import glob

task_name = '/pure_AD_5fold_section_filtered/'

data_dir = './data/'+task_name+'output/fold'


for i in range(5):
  result_f = data_dir+str(i+1)+'/output_update_256/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract/eval_score_test.txt'
  with open(result_f,'r') as f:
    lines = f.readlines()
    for line in lines:
      print (line.strip())
    
    
#output_f = './data/'+task_name
#with open(output_f,'w') as f:
#  f.write('P(exact)\tR(exact)\tF1(exact)\tP(relax)\tR(relax)\tF1(relax)\tright\tright_predict\tright_gold\tpredict\tgold\tSemantic\n')