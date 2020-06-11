#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.


from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import jsonlines as jl
import numpy as np
import os,csv

RESOURCES = [
    DownloadableFile(
        '1FUv2qit9wQ21NV_dbW5HeZVEzng5CMHE',
        'train_self_original.txt',
        '',
        False,
        True
    ),
    DownloadableFile(
        '1lnrgxXCc7Y-6Ic_zl7b3tAXonmuGkjI5',
        'valid_self_original.txt',
        '',
        False,
        True
    )
]

def build_fb_format(q,a,task,dpath):
    if task == 'train':
        N = np.int(len(a)*0.8)
        f = open(os.path.join(dpath, 'train_self_original.txt'), 'w')
        for k in range(2 * N):
            i = k%N
            candindex = np.random.randint(N, size=20).tolist()
            candindex.append(i)
            cand = [a[j] for j in candindex]
            cand = '|'.join(cand)
            sample = str(1) + ' ' + q[i] + '	' + a[i] + '		' + cand + '\n'
            f.write(sample)
        f.close()

    if task == 'valid':
        N1 = len(a)
        N0 = np.int(len(a)*0.8)
        f = open(os.path.join(dpath, 'valid_self_original.txt'), 'w')
        for k in range(N0, N1):
            # i=N0+np.random.randint(N1-N0)
            i = k
            candindex = np.random.randint(N1, size=20).tolist()
            candindex.append(i)
            cand = [a[j] for j in candindex]
            cand = '|'.join(cand)
            sample = str(1) + ' ' + q[i] + '	' + a[i] + '		' + cand + '\n'
            f.write(sample)
        f.close()

def build_download(opt):
    version = 'v1.0'
    dpath = os.path.join(opt['datapath'], 'covid')
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath, check=False)

        # Mark the data as built.
        build_data.mark_done(dpath, version)

def build(opt):
    version = 'v1.0'
    dpath = os.path.join(opt['datapath'], 'covid')
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        dir = '../../../data/scraping/schema_v0.3'
        if not os.path.isdir(dir):
            build_download(opt)
            return

        print('[reading data from: '+dir+']')
        blockID=[]
        with open("../../../data/scraping/blocked_QA_IDs.tsv") as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for line in tsvreader:
                blockID.append(line[0])

        f= open(os.path.join(dpath, 'blockID.txt'), 'w')
        f.write('\n'.join(blockID))
        f.close()

        q = []  # questions
        a = []  # answers
        filelist = []
        for file in os.listdir(dir):
            if file.endswith(".jsonl"):
                filelist.append(os.path.join(dir, file))
        count = 0
        #print(filelist)
        for file in filelist:
            with jl.open(file) as reader:
                for obj in reader:
                    if obj['ID'] in blockID:
                        continue
                    if obj['language'] == 'en':
                        t1=obj['questionText'].replace('\n',' ').replace('\r',' ').replace('\t',' ').replace('  ','')
                        t2=obj['answerText'].replace('\n',' ').replace('\r',' ').replace('\t',' ').replace('  ','')
                        if (len(t1) > 5) & (len(t2) > 5):
                            count += 1
                            q.append(t1)
                            a.append(t2)

        f = open(os.path.join(dpath, 'q.txt'), 'w')
        f.write('\n'.join(q))
        f.close()

        f = open(os.path.join(dpath, 'a.txt'), 'w')
        f.write('\n'.join(a))
        f.close()

        build_fb_format(q, a, 'train', dpath)
        build_fb_format(q, a, 'valid', dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version)


