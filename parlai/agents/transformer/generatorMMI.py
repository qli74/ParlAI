#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# hack to make sure -m transformer/generator works as expected
from .transformer import TransformerGeneratorAgent  # noqa: F401
import torch.nn.functional as F
import torch
from parlai.utils.misc import warn_once
from parlai.core.torch_agent import Batch, Output
from parlai.utils.torch import argsort
from .modules import TransformerGeneratorModel
from copy import deepcopy


class GeneratorMMIAgent(TransformerGeneratorAgent):
    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        #print(self.model._encoder_input(batch))
        if batch.text_vec is None and batch.image is None:
            return Output('N')
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        self.model.eval()
        cand_scores = None
        token_losses = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.label_vec, model_output
                )

        preds = None
        if self.skip_generation:
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning,
            )
        else:
            maxlen = self.label_truncate or 20
            n_best_beam_preds_scores, _ = self._generate(batch, self.beam_size, maxlen)
            preds=[]
            scores=[]
            for n_best_list in n_best_beam_preds_scores:
                p, s = zip(*n_best_list)
                preds.append(p)
                scores.append(s)
        cand_choices = None
        self.rank_candidates=True
        if self.rank_candidates:
            # compute MMI to rank candidates
            bestpreds = []
            for i in range(bsz):
                cands, _ = self._pad_tensor(preds[i])
                cand_scores = self.computeMMI(batch.text_vec[i],cands,list(scores[i]))
                _, ordering = cand_scores.sort()
                bestpreds.append(preds[i][ordering[0]])
        text = [self._v2t(p) for p in bestpreds] if bestpreds is not None else None

        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
            self._compute_nltk_bleu(batch, text)
        return Output(text, cand_choices, token_losses=token_losses)

    def computeMMI(self,source,cands,ptscores):
        num_cands = len(cands)
        max_ts=len(cands[0])
        bsz=1
        w=0.8
        n=0.1
        #print(source.unsqueeze(0))
        encoder_states = self.model.encoder(source.unsqueeze(0))
        p=[]
        #print(cands)
        for c in range(num_cands):
            cand=cands[c]
            decoder_input=cand.unsqueeze(0)
            logits,preds=self.model_inv.decode_forced(encoder_states,decoder_input)
            scores = logits.view(-1, logits.size(-1))
            scores = self.criterion(scores, decoder_input.view(-1))
            #s=sum([logits[0][i][int(cand[i])] for i in range(len(cand))])
            s=-sum(scores)
            p.append((1-w)*s+w*ptscores[c]+n*len(cand.nonzero()[0]))
        return torch.tensor(p)

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt_inv=deepcopy(self.opt)
        #opt_inv['model'] = 'transformer/generatorMMI'
        opt_inv['model_file'] = self.opt['model_file']+'_inv5'
        opt_inv['override']['model_file'] = self.opt['model_file']+'_inv5'
        self.model_inv=TransformerGeneratorModel(opt_inv, self.dict)
        if self.use_cuda:
            self.model_inv.cuda()

