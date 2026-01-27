import time 
import logging

from utils.controller import Controller

class Client:
    def __init__(self, client_monitor, client_res=(576, 1024)):
        self.client_controller = Controller(
            client_monitor,
            window_res=client_res
        )
    
    def start_custom_game(self,
                          blue_side=True,
                          role="mid",
                          champion_name="Orianna",
                          password="",
                          verbose=False):
        self._click_play()
        time.sleep(1.25)
        self._create_custom_game(blue_side, role, password, verbose=verbose)
        time.sleep(1.25)
        self._select_champion(champion_name=champion_name, verbose=verbose)
    
    def _click_play(self):
        self.client_controller.raw_left_click(546, 257)
    
    def _select_champion(self, champion_name, verbose=False):
        # click champion search bar
        self.client_controller.raw_left_click(1092, 310)
        time.sleep(1.25)
        self.client_controller.type(champion_name)
        time.sleep(1.25)
        # click first champion icon
        self.client_controller.raw_left_click(756, 360)
        if verbose:
            logging.info(f"Selected {champion_name}!")
        time.sleep(1.25)
        # click lock champion
        self.client_controller.raw_left_click(961, 710)
        if verbose:
            logging.info(f"Locked in {champion_name}!")

    def _create_custom_game(self,
                            blue_side,
                            role,
                            password,
                            verbose=False):
        if verbose:
            logging.info("Creating a custom game!")
        # click CREATE CUSTOM
        self.client_controller.raw_left_click(758, 307)
        time.sleep(1)
        self._add_password(password, verbose=verbose)
        time.sleep(1)
        self._click_confirm()
        time.sleep(1.5)
        self._select_custom_role(role=role)
        if verbose:
            logging.info(f"Selected {role}!")
        time.sleep(1)
        self._select_custom_side(blue_side=blue_side)
        if verbose:
            logging.info(f"Selected {'blue' if blue_side else 'red'} side!")
        time.sleep(1)
        self._click_confirm()
        if verbose:
            logging.info(f"Started custom game!")

    def _add_password(self, password, verbose=False):
        self.client_controller.raw_left_click(590, 695)
        time.sleep(1)
        self.client_controller.type(password)
        if verbose:
            logging.info(f"Lobby password is '{password}'!")

    def _click_confirm(self):
        self.client_controller.raw_left_click(880, 770)
    
    def _select_custom_role(self, role="mid"):
        self.client_controller.raw_left_click(752, 422)
        time.sleep(1)
        if role == "mid":
            self.client_controller.raw_left_click(787, 502)
        else:
            raise NotImplementedError
    
    def _select_custom_side(self, blue_side=True):
        if not blue_side:
            self.client_controller.raw_left_click(1074, 418)


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
    client = Client(monitor)
    client.start_custom_game(blue_side=False)