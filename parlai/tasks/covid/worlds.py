#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from parlai.core.worlds import create_task
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.tasks.self_chat.worlds import SelfChatBaseWorld

import random
from typing import List

class InteractiveWorld(DialogPartnerWorld):
    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('ConvAI2 Interactive World')


    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.cnt = 0


    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        if self.cnt == 0:
            self.p1, self.p2 = self.get_new_personas()

        acts = self.acts
        human_agent, model_agent = self.agents
        if 0:
            if self.cnt == 0:
                # add the persona on to the first message to human agent
                act = {}
                act['text'] = self.p1
                act['episode_done'] = False
                act['id'] = 'persona'
                human_agent.observe(validate(act))
            act = deepcopy(human_agent.act())
            if self.cnt == 0:
                # add the persona on to the first message to model agent
                act.force_set('text', self.p2 + act.get('text', 'hi'))
                model_agent.observe(validate(act))
            else:
                model_agent.observe(validate(act))
        act = deepcopy(human_agent.act())
        model_agent.observe(validate(act))
        acts[1] = model_agent.act()
        human_agent.observe(validate(acts[1]))
        self.update_counters()
        self.cnt += 1

        if act['episode_done']:
            print("\nCHAT DONE.\n")
            print("[ Preparing new chat ... ]\n")
            self.cnt = 0
