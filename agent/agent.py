import random
import time
import logging
import cv2
import keyboard
import torch

from utils.utils import mouse_id_to_grid_id
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

        self.m_pos = args.m_pos # (n_h, n_w)

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
            (700 - self.agent_controller.w_offset) / self.game_res[1],
            (240 - self.agent_controller.h_offset) / self.game_res[0]
        )
        time.sleep(shop_delay)
        # search for item
        self.agent_controller.type(item_name)
        time.sleep(shop_delay)
        # buy item
        self.agent_controller.left_click(
            (512 - self.agent_controller.w_offset) / self.game_res[1],
            (326 - self.agent_controller.h_offset) / self.game_res[0]
        )
        time.sleep(shop_delay)
        # self.agent_controller.right_click(
        #     (1188 - self.agent_controller.w_offset) / self.game_res[1],
        #     (457 - self.agent_controller.h_offset) / self.game_res[0],
        # )
        self.agent_controller.left_click(
            (1300- self.agent_controller.w_offset) / self.game_res[1],
            (600 - self.agent_controller.h_offset) / self.game_res[0]
        )
        time.sleep(shop_delay)
        # self.agent_controller.left_click(
        #     (1196- self.agent_controller.w_offset) / self.game_res[1],
        #     (580 - self.agent_controller.h_offset) / self.game_res[0]
        # )
        self.agent_controller.left_click(
            (1300- self.agent_controller.w_offset) / self.game_res[1],
            (600 - self.agent_controller.h_offset) / self.game_res[0]
        )
        # exit shop
        self.agent_controller.press_release("p")

    def wait(self, t):
        time.sleep(t)

    def predefined_start(self, game_start_time):
        self.wait(t=4)
        self.buy_item("Doran's Blade", shop_delay=self.args.shop_delay)
        # self.wait(t=14)
        # self.go_mid()
        while self.game.get_game_time() < 15:
            continue

    def collect_trajectory(self, game_start_time, buffer, device, train=True, sample=True):
        self.policy.eval()
        with torch.no_grad():
            while time.time() - game_start_time < self.args.game_end_time:
                self.act(
                    buffer,
                    device, 
                    train=train,
                    sample=sample
                    )
                if keyboard.is_pressed(self.args.stop_key):
                    print("Stopping...")
                    exit()
                time.sleep(self.args.action_delay)

    def act(self, buffer, device, train=True, sample=True):
        game_data, images = self.game.get_game_data()
        
        img = images["game"] # (h, w, 3) img_shape
        img = torch.tensor(img / 255.0).float().permute(2, 0, 1) # (h, w, 3) -> (3, h, w)
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        
        img_b = img.unsqueeze(0).to(device) # add batch dimension (3, h, w) -> (1, 3, h, w)
        value, action, logp = self.policy.sample_action(img_b) if sample else self.policy.take_best_action(img_b)
        # logging.info(f"{action[0]}, {action[1:]}, {logp}")
        
        # execute action
        # if action[0]:
        #     self._move_click(*action[1:])
        self._execute_action(action)
        
        if train:
            # reward = reward_model.get_reward()
            game_data, images = self.game.get_game_data()
            buffer.add(
                img,
                game_data,
                action,
                0,
                logp,
                value
                )

    def random_action(self, img=None):
        self._move_click(random.random(), random.random())
    
    def _move_click(self, x, y):# TODO move this to AgentController(Controller) class
        self.agent_controller.right_click(x, y)
        # logging.info(f"Click ({x}, {y})")
    
    def _execute_action(self, action):
        grid_x, grid_y = mouse_id_to_grid_id(action[1], self.m_pos)
        x = (grid_x + 0.5) / self.m_pos[1]
        y = (grid_y + 0.5) / self.m_pos[0]
        self.agent_controller.mouse_to(x, y)
        if action[0]:
            self.agent_controller.only_right_click()
