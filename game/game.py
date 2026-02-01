import time
import requests
import urllib.parse, urllib3
import logging
import cv2
from pynput.keyboard import Key

from utils.controller import Controller
from utils.screen import Frame

class Game:
    def __init__(self, sct, game_monitor, game_res=(720, 1280)):
        self.game_controller = Controller(
            game_monitor,
            window_res=game_res
        )

        self.base_url = "https://127.0.0.1:2999/liveclientdata/"

        self.frame = Frame(sct, game_monitor, frame_res=game_res)
        self.load_flag = False
        self.start_flag = True
    
    def capture_frame(self, shape=None, save=""):
        return self.frame.capture_game_frame(shape=shape, save=save)
    
    def capture_minimap(self, shape=None, save=""):
        return self.frame.capture_minimap(shape=shape, save=save)

    def enter_game(self):
        time.sleep(2)
        self.game_controller.left_click(0.95, 0.05)
        time.sleep(0.5)

    def leave_game(self, verbose=False):
        self.game_controller.press_release(Key.esc)
        time.sleep(0.5)
        self.game_controller.left_click(
            (734 - self.game_controller.w_offset) / self.game_controller.game_res[1],
            (742 - self.game_controller.h_offset) / self.game_controller.game_res[0]
        )
        time.sleep(1)
        self.game_controller.left_click(
            (880 - self.game_controller.w_offset) / self.game_controller.game_res[1],
            (500 - self.game_controller.h_offset) / self.game_controller.game_res[0]
        )
        if verbose:
            logging.info("Left the game!")
        time.sleep(2)
    
    def wait_game_start(self, verbose=False):
        if verbose:
            logging.info("Waiting for game to start...")
        time.sleep(13)
        self.load_flag = False
        self.start_flag = True
        t = time.time()
        while not self.game_started(verbose=verbose):
            if time.time() - t > 120:
                logging.info("Failed to start game")
                exit()
        if verbose:
            logging.info("Game has started!")

    def game_started(self, verbose=False):
        try:
            self.get_game_time(verbose=verbose)
            return True
        except TypeError:
            return False
        except:
            raise NotImplementedError

    def get_player_is_alive(self):
        return not self.get_live_game_data()["allPlayers"][0]["isDead"]

    def get_player_level(self):
        return self.get_live_game_data(endpoint="activeplayer")["level"]

    def get_player_gold(self):
        return float(self.get_live_game_data(endpoint="activeplayer")["currentGold"])
    
    def get_game_time(self, verbose=False):
        return self.get_live_game_data(endpoint="gamestats", verbose=verbose)["gameTime"]

    def get_live_game_data(self, endpoint="allgamedata", verbose=False):
        url = self.base_url + endpoint
        try:
            # We replace verify=False with the path to the cert
            # response = requests.get(url, verify=self.cert_path, timeout=5)
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(url, timeout=0.3, verify=False)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                if not self.load_flag and verbose:
                    logging.info("Currently in loading screen!")
                    self.load_flag = True
            else:
                logging.warning(f"Server returned status code: {response.status_code}")
                
        except requests.exceptions.SSLError:
            logging.error("SSL Verification failed! Check if 'riotgames.pem' is correct.")
        except requests.exceptions.ConnectionError:
            if not self.start_flag and verbose:
                logging.info("Waiting for game to start!")
                self.start_flag = True
        return None

if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log", mode="w"),  # Save logs to file
        logging.StreamHandler()               # Print to console
    ]
    )
    import mss
    sct = mss.mss()
    monitor = sct.monitors[1]
    game = Game(sct, monitor)
    game.wait_game_start()
    print(game.get_game_time())
    game.capture_frame("test.png")
