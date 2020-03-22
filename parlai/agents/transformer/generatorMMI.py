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

class GeneratorMMIAgent(TransformerGeneratorAgent):
    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        print(batch)
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
                cand_scores = self.computeMMI(batch.text_vec[i],cands)
                _, ordering = cand_scores.sort()
                bestpreds.append(preds[i][ordering[0]])
        text = [self._v2t(p) for p in bestpreds] if bestpreds is not None else None

        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
            self._compute_nltk_bleu(batch, text)
        return Output(text, cand_choices, token_losses=token_losses)



    def computeMMI(self,source,cands):
        num_cands = len(cands)
        max_ts=len(cands[0])
        bsz=1
        #print(source.unsqueeze(0))
        encoder_states = self.model.encoder(source.unsqueeze(0))
        decoder_input = (
            torch.LongTensor([self.START_IDX]).expand(bsz, 1)
        )
        p=[]
        for c in range(num_cands):
            cand=cands[c]
            incr_state = None
            scores, incr_state = self.model.decoder(decoder_input, encoder_states, incr_state)
            for _ts in range(max_ts-1):
                s, incr_state = self.model.decoder(cand[_ts].view(bsz,-1), encoder_states, incr_state)
                s = F.log_softmax(s, dim=-1)
                scores=torch.cat((scores,s),1)
            p.append(sum([scores[0][i][int(cand[i])] for i in range(len(cand))]))
        return torch.tensor(p)

