import random
import time
import logging
import torch

from utils.utils import inference_mode
from utils.controller import Controller
from agent.policy.policy import Policy
from ppo.buffer import RolloutBufferDataset
from ppo.reward import RewardModel


class Agent:
    def __init__(self, game_monitor, game, policy, args):
        self.args = args
        self.game_res = args.game_res

        self.agent_controller = Controller(
            game_monitor,
            window_res=self.game_res
        )
        self.game = game
        self.policy = policy

    def go_mid(self):
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

    def predefined_start(self, game_start_time):
        self.wait(t=1)
        self.buy_item("Doran's Blade", shop_delay=self.args.shop_delay)
        self.wait(t=14)
        self.go_mid()
        while time.time() - game_start_time < self.args.minions_time:
            continue

    def collect_trajectory(self, game_start_time, buffer, reward_model, device, train=True):
        with inference_mode(self.policy):
            while time.time() - game_start_time < self.args.game_end_time:
                self.act(
                    buffer,
                    reward_model,
                    device, 
                    img_shape=self.args.img_shape,
                    train=train
                    )
                time.sleep(self.args.action_delay)

    def act(self, buffer, reward_model, device, img_shape=(256, 256), train=True):
        img = torch.tensor(self.game.capture_frame(shape=img_shape) / 255.0).permute(2, 0, 1).float()
        img_b = img.unsqueeze(0).to(device)
        value, action, logp = self.policy.sample_action(img_b)
        # logging.info(f"{action[0]}, {action[1:]}, {logp}")
        
        # execute action
        if action[0]:
            self._move_click(*action[1:])
        
        if train:
            reward = reward_model.get_reward()
            buffer.add(
                img,
                action,
                reward,
                logp,
                value
                )

    def random_action(self, img=None):
        self._move_click(random.random(), random.random())
    
    def _move_click(self, x, y):# TODO move this to AgentController(Controller) class
        self.agent_controller.right_click(x, y)
        # logging.info(f"Click ({x}, {y})")
