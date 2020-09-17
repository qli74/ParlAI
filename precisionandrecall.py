#!/usr/bin/env python3
import subprocess
import json
import numpy as np
#data/models/pretrained_transformers/poly_model_huge_reddit/model
#model/model/poly/covid7
for thre in np.arange(1, 30, 1):
    p = subprocess.Popen(['python3 -m parlai.scripts.interactive -m transformer/polyencoder -mf ../model/covid --single-turn True --inference topp --topp '+str(thre)+' --ground-truth-path ../ROCdata/testQA.tsv'], shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE,
                        env={})

    for i in range(200):
        line = p.stdout.readline()
        line = line.strip().decode()
        print(line)
        if line == '[  polyencoder_type: codes ]':
            break

    precision=[]
    recall=[]
    f1=[]
    for q in range(254):
        q=str(q)
        print(q)
        p.stdin.write(q.encode('utf-8'))
        #p.stdin.write("\n")
        p.stdin.write(b"\n")
        p.stdin.flush()
        line=''
        while 'precision:' not in line: # Exclude other messages
            line = p.stdout.readline()
            line = line.strip().decode()
            print(line)
        line = line.split('precision:')
        pre=float(line[-1])

        line = p.stdout.readline()
        line = line.strip().decode()
        print(line)
        line = line.split('recall:')
        rec = float(line[-1])

        precision.append(pre)
        recall.append(rec)
        f=2*pre*rec/(pre+rec)
        f1.append(f)


    print(np.mean(precision),np.mean(recall),np.mean(f1))
    p.kill()
    f = open("pr.txt","a")
    f.write(str(thre)+' '+str(np.mean(precision))+' '+str(np.mean(recall))+' '+str(np.mean(f1))+'\n')
    f.close()
