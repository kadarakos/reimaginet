from driver import *
import os
import sys
import numpy as np

def load(f):
        return pickle.load(gzip.open(os.path.join(f)))

if len(sys.argv) < 2:
    print "\nCall me with python interface.py train/test/predict"
    sys.exit()

    
    
if sys.argv[1] == 'train':
    project_name = raw_input("Create new project in directory ? >> ")
    os.mkdir(project_name)
    cmd_train( dataset='coco',
                   datapath='/home/akadar/pymodules/bravelearner/',
                   model_path=project_name+'/',
                   hidden_size=1024,
                   gru_activation=clipped_rectify,
                   visual_activation=linear,
                   max_norm=None,
                   embedding_size=None,
                   depth=1,
                   scaler=None,
                   cost_visual=CosineDistance,
                   seed=None,
                   shuffle=True,
                   with_para='auto',
                   architecture=MultitaskLM,
                   dropout_prob=0.0,
                   alpha=0.1,
                   epochs=7,
                   batch_size=16,
                   validate_period=64*10000,
                   logfile='log.txt')
    
elif sys.argv[1] == 'test':
    project_name = raw_input("Read models from directory ? >> ")
    models = filter(lambda x: x.split('.')[0] == 'model',  os.listdir(project_name))
    print "Models", models
    for model_name in models:
        print "Predicting \n"
	print model_name
        cmd_predict_v(dataset='coco',
                      datapath='/home/akadar/pymodules/bravelearner/',
                      model_path=project_name,
                      model_name=model_name,
                      batch_size=64,
                      output_v='predict_v.npy',
                      output_r='predict_r.npy')

        print "Evaluating \n"
	print model_name
        cmd_eval(dataset='coco',
                     datapath='/home/akadar/pymodules/bravelearner/',
                     scaler_path=project_name+'/scaler.pkl.gz',
                     input_v=project_name+'/predict_v.npy',
                     input_r=project_name+'/predict_r.npy',
                     output=project_name+'/eval.json')
        
elif sys.argv[1] == 'predict':
    project_name = './'+raw_input("Directory of the model ? >> ")+'/'
    
    print os.listdir(project_name), '\n'
    model_name   = raw_input("Name of the model ? >> ")
    data_path    = raw_input("Path to sentences ? >> ")
    batcher, scaler, model = map(load, 
                                 [project_name+'batcher.pkl.gz',
                                  project_name+'scaler.pkl.gz',
                                 project_name+model_name
                                 ])
    mapper = batcher.mapper
    predict_v = predictor_v(model)
    predict_r = predictor_r(model)
    sents = open(data_path,'r').read().split('\n')
    sents = map(str.split, sents)
    #inputs = list(mapper.transform([tokens(sent, tokenizer=batcher.tokenizer) for sent in sents ]))
    inputs = list(mapper.transform(sents))
    preds_v = []
    preds_r = []
    for j, i in enumerate(inputs):
        if i != []:
            preds_v.append( predict_v([i])[0] )
            preds_r.append( predict_r([i])[0] )
    numpy.save(os.path.join(project_name, 'predict_v.npy'), 
               np.vstack(preds_v))  
    

    numpy.save(os.path.join(project_name, 'predict_r.npy'),
               np.vstack(preds_r))
    

        
 
