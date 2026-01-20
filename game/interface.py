import mss
import time, random, cv2
import logging
from pynput import keyboard

from game.client import Client
from game.game import Game


class Interface:
    def __init__(self, sct, monitor, args):
        self.args = args

        self.client = Client(monitor, client_res=args.client_res)
        self.game = Game(sct, monitor, game_res=args.game_res)
        self.agent = Agent(monitor, game_res=args.game_res)
    
    def train_agent(self):
        for i in range(self.args.train_epochs):
            self.collect_trajectory()
            

    def collect_trajectory(self, blue_side=random.random()>0.5):
        self.client.start_custom_game(
            blue_side=blue_side,
            champion_name="Master Yi",
            password=self.args.password,
            verbose=self.args.verbose)
        self.game.wait_game_start(verbose=self.args.verbose)
        self.game.enter_game()

        self.agent.buy_item("Doran's Blade", shop_delay=self.args.shop_delay)
        self.agent.wait(t=15)
        self.agent.go_mid(blue_side)
        while self.game.get_game_time() < 48:
            continue
        while self.game.get_game_time() < self.args.game_end_time:
            self.agent.act(
                self.game,
                img_shape=self.args.img_shape,
                train=True
                )
            time.sleep(self.args.action_delay)
            # self.agent.random_action()
        self.game.leave_game(verbose=self.args.verbose)

    def test(self):
        self.game.enter_game()
        self.agent.buy_item("Doran's Ring")

    def run_custom_game(self, blue_side=random.random()>0.5):
        self.client.start_custom_game(
            blue_side=blue_side,
            champion_name="Master Yi",
            password=self.args.password,
            verbose=self.args.verbose)
        self.game.wait_game_start(verbose=self.args.verbose)
        self.game.enter_game()

        self.agent.buy_item("Doran's Blade", shop_delay=self.args.shop_delay)
        self.agent.wait(t=15)
        self.agent.go_mid(blue_side)
        while self.game.get_game_time() < 48:
            continue
        while self.game.get_game_time() < self.args.game_end_time:
            self.agent.random_action()
        self.game.leave_game(verbose=self.args.verbose)

    def start(self, stop_char="l"):
        listener = keyboard.Listener(on_press=lambda key: self._on_key_press(key, stop_char=stop_char))
        listener.start()
        
    def _on_key_press(self, key, stop_char="l"):
        if key.char == stop_char:
            logging.warning(f"Stopping loop ({stop_char} pressed)")
            exit()

    def test_loop(self):
        self.start(stop_char="l")
        verbose = False
        self.client.start_custom_game(password=self.args.password, verbose=verbose)
        self.game.wait_game_start(verbose=verbose)
        self.game.enter_game()
        self.game.leave_game(verbose=verbose)
            