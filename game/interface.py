import mss
import time, random, cv2
import logging
from pynput import keyboard

from game.client import Client
from game.game import Game


class Interface:
    def __init__(self, client, game, args):
        self.args = args

        self.client = client
        self.game = game        

    def start_custom_game(self, blue_side):
        self.client.start_custom_game(
            blue_side=blue_side,
            champion_name="Miss Fortune",
            password=self.args.password,
            verbose=self.args.verbose)
        self.game.wait_game_start(verbose=self.args.verbose)
        game_start_time = time.time()
        self.game.enter_game()
        return game_start_time
    
    def end_custom_game(self):
        self.game.leave_game(verbose=self.args.verbose)

    def start(self, stop_char="l"):
        listener = keyboard.Listener(on_press=lambda key: self._on_key_press(key, stop_char=stop_char))
        listener.start()
        
    def _on_key_press(self, key, stop_char="l"):
        if key.char == stop_char:
            logging.warning(f"Stopping loop ({stop_char} pressed)")
            exit()
            