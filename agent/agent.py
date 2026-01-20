import random
import time
import logging
import torch

from utils.controller import Controller
from agent.policy.policy import Policy
from data.buffer import RolloutBuffer
from ppo.reward import Reward


class Agent:
    def __init__(self, game_monitor, policy, args, device="cuda"):
        self.args = args
        self.game_res = args.game_res

        self.agent_controller = Controller(
            game_monitor,
            window_res=self.game_res
        )
        

        self.actions = ["nothing", "move_click"]
        self.action_specs = {"move_click": 2}

        self.device = device

        self.policy = Policy(
            actions=self.actions,
            action_specs=self.action_specs
            )
        
        self.buffer = RolloutBuffer()
        self.reward = Reward()

    def go_mid(self, blue_side):
        # if blue_side:
        #     self._move_click(
        #         (1524 - self.agent_controller.w_offset) / self.game_res[1],
        #         (833 - self.agent_controller.h_offset) / self.game_res[0]
        #         )
        # else:
        #     self._move_click(
        #         (1537 - self.agent_controller.w_offset) / self.game_res[1],
        #         (826 - self.agent_controller.h_offset) / self.game_res[0]
        #         )
        self._move_click(
                (1497 - self.agent_controller.w_offset) / self.game_res[1],
                (797 - self.agent_controller.h_offset) / self.game_res[0]
                )

    def buy_item(self, item_name, shop_delay=0.05):
        # TODO: implement gold check, multiple items
        # open shop
        self.agent_controller.press_release("p")
        time.sleep(shop_delay)
        # click shop search
        self.agent_controller.left_click(
            (827 - self.agent_controller.w_offset) / self.game_res[1],
            (348 - self.agent_controller.h_offset) / self.game_res[0]
        )
        time.sleep(shop_delay)
        # search for item
        self.agent_controller.type(item_name)
        time.sleep(shop_delay)
        # buy item
        self.agent_controller.left_click(
            (700 - self.agent_controller.w_offset) / self.game_res[1],
            (400 - self.agent_controller.h_offset) / self.game_res[0]
        )
        time.sleep(shop_delay)
        # self.agent_controller.right_click(
        #     (1188 - self.agent_controller.w_offset) / self.game_res[1],
        #     (457 - self.agent_controller.h_offset) / self.game_res[0],
        # )
        self.agent_controller.left_click(
            (1196- self.agent_controller.w_offset) / self.game_res[1],
            (580 - self.agent_controller.h_offset) / self.game_res[0]
        )
        time.sleep(shop_delay)
        self.agent_controller.left_click(
            (1196- self.agent_controller.w_offset) / self.game_res[1],
            (580 - self.agent_controller.h_offset) / self.game_res[0]
        )
        # exit shop
        self.agent_controller.press_release("p")

    def wait(self, t):
        time.sleep(t)

    def act(self, game, img_shape=(256, 256), train=True):
        img = game.capture_frame(shape=img_shape) / 255.0
        img_b = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        value, action, logp = self.policy.sample_action(img_b)
        # logging.info(f"{action[0]}, {action[1:]}, {logp}")

        # execute action
        if action[0]:
            self._move_click(*action[1:])

        reward = self.reward.get(game)
        
        if train:
            self.buffer.add(
                img,
                action,
                logp,
                reward
                )

    def random_action(self, img=None):
        self._move_click(random.random(), random.random())
    
    def _move_click(self, x, y):# TODO move this to AgentController(Controller) class
        self.agent_controller.right_click(x, y)
        # logging.info(f"Click ({x}, {y})")
