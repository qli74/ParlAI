#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.


from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os

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


def build_fb_format():
    pass


def build(opt):
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
            downloadable_file.download_file(dpath,check=False)

        # Mark the data as built.
        build_data.mark_done(dpath, version)

    dpath = os.path.join(opt['datapath'][:-5], 'model')
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    if not build_data.built(dpath, version):
        print('[downloading model: covid7]')
        # Download the model
        model=DownloadableFile(
            '13rbP7bxj7Pq412ULUlvgCCIweSzJTwJF',
            'poly.zip',
            '',
            True,
            True
        )
        model.download_file(dpath, check=False)

        # Mark the data as built.
        build_data.mark_done(dpath, version)

