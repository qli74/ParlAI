from parlai.core.torch_generator_agent import TorchGeneratorAgent, Output
from .modules import *
import argparse,time
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init
from collections import Counter
from torch.utils.data import Dataset

class HredAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-n', dest='name', help='enter suffix for model files', default='full_hc')
        agent.add_argument('-e', dest='epoch', type=int, default=20, help='number of epochs')
        agent.add_argument('-pat', dest='patience', type=int, default=-1,
                            help='validtion patience for early stopping default none')
        agent.add_argument('-tc', dest='teacher', action='store_true', default=False, help='default teacher forcing')
        agent.add_argument('-bi', dest='bidi', action='store_true', default=False, help='bidirectional enc/decs')
        agent.add_argument('-test', dest='test', action='store_true', default=False, help='only test or inference')
        agent.add_argument('-shrd_dec_emb', dest='shrd_dec_emb', action='store_true', default=False,
                            help='shared embedding in/out for decoder')
        agent.add_argument('-btstrp', dest='btstrp', default=None, help='bootstrap/load parameters give name')
        agent.add_argument('-lm', dest='lm', action='store_true', default=False,
                            help='enable a RNN language model joint training as well')
        agent.add_argument('-toy', dest='toy', action='store_true', default=False,
                            help='loads only 1000 training and 100 valid for testing')
        agent.add_argument('-pretty', dest='pretty', action='store_true', default=False, help='pretty print inference')
        agent.add_argument('-mmi', dest='mmi', action='store_true', default=False,
                            help='Using the mmi anti-lm for ranking beam')
        agent.add_argument('-drp', dest='drp', type=float, default=0.3, help='dropout probability used all throughout')
        agent.add_argument('-nl', dest='num_lyr', type=int, default=1, help='number of enc/dec layers(same for both)')
        agent.add_argument('-bms', dest='beam', type=int, default=1, help='beam size for decoding')
        agent.add_argument('-vsz', dest='vocab_size', type=int, default=10004, help='size of vocabulary')
        agent.add_argument('-esz', dest='emb_size', type=int, default=300, help='embedding size enc/dec same')
        agent.add_argument('-uthid', dest='ut_hid_size', type=int, default=600, help='encoder utterance hidden state')
        agent.add_argument('-seshid', dest='ses_hid_size', type=int, default=1200, help='encoder session hidden state')
        agent.add_argument('-dechid', dest='dec_hid_size', type=int, default=600, help='decoder hidden state')

        super(HredAgent, cls).add_cmdline_args(argparser)
        return agent


    def __init__(self, opt, shared=None):
        """
        Set up model.
        """
        super().__init__(opt, shared)
        self.id = 'hred'

    def train_step(self,batch):
        self.is_training=True
        if len(self.history.history_vecs)<2:
            #print('no history, skip this batch')
            return
        #print(batch)
        options=self.opt_obj

        #print(self.history.history_vecs,self.history.history_strings )
        with open(options.override['dict_file']+'.pkl', 'rb') as fp2:
            dict_data = pickle.load(fp2)

        options=self.opt_obj
        model=self.model
        model.train()
        self.optimizer = optim.Adam(model.parameters(), options.learningrate)
        #if options.btstrp:
        #    self.load_model_state(self.model, options.btstrp + "_mdl.pth")
        #    self.load_model_state(self.optimizer, options.btstrp + "_opti_st.pth")
        #else:
        #    self.init_param()

        criteria = nn.CrossEntropyLoss(ignore_index=10003, reduction='sum' )
        if use_cuda:
            criteria.cuda()

        best_vl_loss, patience, batch_id = 10000, 0, 0

        tr_loss, tlm_loss, num_words = 0, 0, 0
        strt = time.time()
        #train batch
        u1=sent_to_tensor(dict_data,self.history.history_strings[-2]).unsqueeze(0)
        u2=sent_to_tensor(dict_data,batch['observations'][0]['text']).unsqueeze(0)
        u3=sent_to_tensor(dict_data,batch['labels'][0]).unsqueeze(0)
        #sample_batch=custom_collate_fn(u1,u2,u3,options.batchsize)
        sample_batch=(u1,[min(len(u1),10003)],u2,[min(len(u2),10003)],u3,[min(len(u3),10003)])
        #print(sample_batch)
        new_tc_ratio = 2100.0 / (2100.0 + math.exp(batch_id / 2100.0))
        model.dec.set_tc_ratio(new_tc_ratio)

        preds, lmpreds = model(sample_batch)
        #print(preds,lmpreds)
        u3 = sample_batch[4]
        if use_cuda:
            u3 = u3.cuda()

        preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
        u3 = u3[:, 1:].contiguous().view(-1)

        loss = criteria(preds, u3)
        #print(u3.ne(10003).long().sum().data)
        target_toks = u3.ne(10003).long().sum().item()

        num_words += target_toks
        tr_loss += loss.item()
        loss = loss / target_toks

        if options.lm:
            lmpreds = lmpreds[:, :-1, :].contiguous().view(-1, lmpreds.size(2))
            lm_loss = criteria(lmpreds, u3)
            tlm_loss += lm_loss.item()
            lm_loss = lm_loss / target_toks

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if options.lm:
            lm_loss.backward()
        self.clip_gnorm()
        self.optimizer.step()
        self.metrics['loss']=tr_loss / num_words
        #print('training loss: ',tr_loss / num_words)
        self.batch_id += 1
        self.update_params()

    def eval_step(self,batch):
        # for each row in batch, convert tensor to back to text strings
        self.inference_beam(batch)
        answer = self.uniq_answer()
        answer = answer.replace('<s> ', '').replace(' </s>', '')
        return Output(text=[answer])

    def build_model(self):
        """
        Initialize model.
        """
        opt=self.opt
        # convert dictionary to object
        self.opt_obj = Option_object(**opt)
        self.batch_id=0
        model = Seq2Seq(self.opt_obj)
        return model

    def clip_gnorm(self):
        for name, p in self.model.named_parameters():
            param_norm = p.grad.data.norm()
            if param_norm > 1:
                p.grad.data.mul_(1 / param_norm)

    #def load_model_state(self,mdl, fl):
    #    saved_state = torch.load(fl)
    #    mdl.load_state_dict(saved_state)

    def init_param(self):
        for name, param in self.model.named_parameters():
            # skip over the embeddings so that the padding index ones are 0
            if 'embed' in name:
                continue
            elif ('rnn' in name or 'lm' in name) and len(param.size()) >= 2:
                init.orthogonal_(param)
            else:
                init.normal_(param, 0, 0.01)

    def calc_valid_loss(self,data_loader, criteria):
        self.model.eval()
        cur_tc = self.model.dec.get_teacher_forcing()
        self.model.dec.set_teacher_forcing(True)
        # we want to find the perplexity or likelihood of the provided sequence

        valid_loss, num_words = 0, 0
        for i_batch, sample_batch in enumerate(tqdm(data_loader)):
            preds, lmpreds = self.model(sample_batch)
            u3 = sample_batch[4]
            if use_cuda:
                u3 = u3.cuda()
            preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
            u3 = u3[:, 1:].contiguous().view(-1)
            # do not include the lM loss, exp(loss) is perplexity
            loss = criteria(preds, u3)
            num_words += u3.ne(10003).long().sum().item()
            valid_loss += loss.item()

        self.model.train()
        self.model.dec.set_teacher_forcing(cur_tc)

        return valid_loss / num_words


    def inference_beam(self,batch):
        #print(self.opt)
        options=self.opt_obj
        #print(options.override['dict_file'])
        with open(options.override['dict_file']+'.pkl', 'rb') as fp2:
            dict_data = pickle.load(fp2)
        with open(options.override['dict_file']+'_inv.pkl', 'rb') as fp2:
            inv_dict = pickle.load(fp2)

        criteria = nn.CrossEntropyLoss(ignore_index=10003, reduction='sum')
        if use_cuda:
            criteria.cuda()
        cur_tc = self.model.dec.get_teacher_forcing()
        self.model.dec.set_teacher_forcing(True)
        fout = open(options.name + "_result.txt", 'w')
        #self.load_model_state(self.model, options.name + "_mdl.pth")
        self.model.eval()

        #test_ppl = self.calc_valid_loss(dataloader, criteria)
        #print("test preplexity is:{}".format(test_ppl))
        #print(self.history.history_strings)
        u2 = sent_to_tensor(dict_data, batch['observations'][0]['text']).unsqueeze(0)

        try:
            u1=sent_to_tensor(dict_data, self.history.history_strings[-2]).unsqueeze(0)
        except:
            u1 = torch.LongTensor([[1,2]])
        if batch['label_vec'] is None:
            u3 = torch.LongTensor([[1,2]])
        else:
            u3 = sent_to_tensor(dict_data, batch['labels'][0]).unsqueeze(0)

        # sample_batch=custom_collate_fn(u1,u2,u3,options.batchsize)
        sample_batch = (u1, [min(len(u1),10003)], u2, [min(len(u2),10003)], u3, [min(len(u3),10003)])

        u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], sample_batch[3], \
                                                sample_batch[4], sample_batch[5]

        if use_cuda:
            u1 = u1.cuda()
            u2 = u2.cuda()
            u3 = u3.cuda()
        #(u1,u2,u3)
        o1, o2 = self.model.base_enc((u1, u1_lens)), self.model.base_enc((u2, u2_lens))
        qu_seq = torch.cat((o1, o2), 1)
        # if we need to decode the intermediate queries we may need the hidden states
        final_session_o = self.model.ses_enc(qu_seq)
        # forward(self, ses_encoding, x=None, x_lens=None, beam=5 ):
        for k in range(options.batchsize):
            sent = self.generate(final_session_o[k, :, :].unsqueeze(0), options)
            pt = tensor_to_sent(sent, inv_dict)
            #print(pt)
            # greedy true for below because only beam generates a tuple of sequence and probability
            gt = tensor_to_sent(u3[k, :].unsqueeze(0).data.cpu().numpy(), inv_dict, True)
            fout.write(str(gt[0]) + "    |    " + str(pt[0][0]) + "\n")
            fout.flush()

            #if not options.pretty:
                #print(pt)
                #print("Ground truth {} {} \n".format(gt, self.get_sent_ll(u3[k, :].unsqueeze(0), u3_lens[k:k + 1],
            #                                                         criteria, final_session_o)))
            #else:
                #print(gt[0], "|", pt[0][0])

        self.model.dec.set_teacher_forcing(cur_tc)
        fout.close()

    def generate(self,ses_encoding, options):
        diversity_rate = 2
        antilm_param = 10
        beam = options.beam

        n_candidates, final_candids = [], []
        candidates = [([1], 0, 0)]
        gen_len, max_gen_len = 1, 20

        # we provide the top k options/target defined each time
        while gen_len <= max_gen_len:
            for c in candidates:
                seq, pts_score, pt_score = c[0], c[1], c[2]
                with torch.no_grad():
                    _target = Variable(torch.LongTensor([seq]))
                dec_o, dec_lm = self.model.dec([ses_encoding, _target, [len(seq)]])
                dec_o = dec_o[:, :, :-1]

                op = F.log_softmax(dec_o, 2, 5)
                op = op[:, -1, :]
                topval, topind = op.topk(beam, 1)

                if options.lm:
                    dec_lm = dec_lm[:, :, :-1]
                    lm_op = F.log_softmax(dec_lm, 2, 5)
                    lm_op = lm_op[:, -1, :]

                for i in range(beam):
                    ctok, cval = topind.data[0, i], topval.data[0, i]
                    if options.lm:
                        uval = lm_op.data[0, ctok]
                        if dec_lm.size(1) > antilm_param:
                            uval = 0.0
                    else:
                        uval = 0.0

                    if ctok == 2:
                        list_to_append = final_candids
                    else:
                        list_to_append = n_candidates

                    list_to_append.append((seq + [ctok], pts_score + cval - diversity_rate * (i + 1), pt_score + uval))

            n_candidates.sort(key=lambda temp: sort_key(temp, options.mmi), reverse=True)
            candidates = copy.copy(n_candidates[:beam])
            n_candidates[:] = []
            gen_len += 1

        final_candids = final_candids + candidates
        final_candids.sort(key=lambda temp: sort_key(temp, options.mmi), reverse=True)

        return final_candids[:beam]

    def get_sent_ll(self,u3, u3_lens, criteria, ses_encoding):
        preds, _ = self.model.dec([ses_encoding, u3, u3_lens])
        preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
        u3 = u3[:, 1:].contiguous().view(-1)
        loss = criteria(preds, u3).item()
        target_toks = u3.ne(10003).long().sum().item()
        return -1 * loss / target_toks

    def uniq_answer(self):
        fil=self.opt_obj.name
        uniq = Counter()
        with open(fil + '_result.txt', 'r') as fp:
            all_lines = fp.readlines()
            for line in all_lines:
                resp = line.split("    |    ")
                uniq[resp[1].strip()] += 1
        #print('uniq', len(uniq), 'from', len(all_lines))
        #print('---all---')
        answers=uniq.most_common()
        #print('num of answer:',len(answers))
        #print(answers)
        return answers[0][0]
