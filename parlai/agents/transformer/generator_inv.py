#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# hack to make sure -m transformer/generator works as expected
from .transformer import TransformerGeneratorAgent # noqa: F401
from parlai.core.metrics import SumMetric
from parlai.utils.misc import warn_once
from parlai.core.torch_agent import Batch, Output

class GeneratorInvAgent(TransformerGeneratorAgent):
    def train_step(self, batch):
        """
        Train on a single batch of examples.
        """
        # helps with memory usage
        # note we want to use the opt's batchsize instead of the observed batch size
        # in case dynamic batching is in use
        batch['text_vec'],batch['label_vec']=batch['label_vec'],batch['text_vec']
        batch['text_lengths'], batch['label_lengths'] = batch['label_lengths'], batch['text_lengths']
        self._init_cuda_buffer(self.opt['batchsize'], self.label_truncate or 256)
        self.model.train()
        self.zero_grad()

        try:
            loss = self.compute_loss(batch)
            self.backward(loss)
            self.update_params()
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print(
                    '| WARNING: ran out of memory, skipping batch. '
                    'if this happens frequently, decrease batchsize or '
                    'truncate the inputs to the model.'
                )
                self.global_metrics.update('skipped_batches', SumMetric(1))
                # gradients are synced on backward, now this model is going to be
                # out of sync! catch up with the other workers
                self._init_cuda_buffer(8, 8, True)
            else:
                raise e

        def eval_step(self, batch):
            """
            Evaluate a single batch of examples.
            """
            if batch.text_vec is None and batch.image is None:
                return
            batch['text_vec'], batch['label_vec'] = batch['label_vec'], batch['text_vec']
            batch['text_lengths'], batch['label_lengths'] = batch['label_lengths'], batch['text_lengths']

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
                maxlen = self.label_truncate or 256
                beam_preds_scores, _ = self._generate(batch, self.beam_size, maxlen)
                preds, scores = zip(*beam_preds_scores)

            cand_choices = None
            # TODO: abstract out the scoring here

            if self.rank_candidates:
                # compute roughly ppl to rank candidates
                cand_choices = []
                encoder_states = self.model.encoder(*self._encoder_input(batch))
                for i in range(bsz):
                    num_cands = len(batch.candidate_vecs[i])
                    enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                    cands, _ = self._pad_tensor(batch.candidate_vecs[i])
                    scores, _ = self.model.decode_forced(enc, cands)
                    cand_losses = F.cross_entropy(
                        scores.view(num_cands * cands.size(1), -1),
                        cands.view(-1),
                        reduction='none',
                    ).view(num_cands, cands.size(1))
                    # now cand_losses is cands x seqlen size, but we still need to
                    # check padding and such
                    mask = (cands != self.NULL_IDX).float()
                    cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                    _, ordering = cand_scores.sort()
                    cand_choices.append([batch.candidates[i][o] for o in ordering])

            text = [self._v2t(p) for p in preds] if preds is not None else None
            if text and self.compute_tokenized_bleu:
                # compute additional bleu scores
                self._compute_fairseq_bleu(batch, preds)
                self._compute_nltk_bleu(batch, text)
            return Output(text, cand_choices, token_losses=token_losses)