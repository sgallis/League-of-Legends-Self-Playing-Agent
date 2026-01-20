import argparse
from pynput.mouse import Controller, Button
import time
import random
import logging

from game.interface import Interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor_idx", type=int, default=1)
    args = parser.parse_args()
    args.game_res = (720, 1280)
    args.client_res = (576, 1024)
    args.password = "e2h8nef87*"
    args.verbose = True
    args.game_end_time = 100
    args.shop_delay = 0.15
    args.device = "cuda"
    args.img_shape=(256, 256)
    args.action_delay = 0.5
    args.train_epochs = 3

    logging.basicConfig(
    level=logging.INFO,  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log", mode="w"),  # Save logs to file
        logging.StreamHandler()               # Print to console
    ]
    )

    interface = Interface(args)
    # interface.game.capture_frame(args.img_shape, save="test.png")
    interface.collect_trajectory()
    # while True:
    #     time.sleep(1)
    #     # interface.agent.reward.get(interface.game)
    #     mouse = Controller()
    #     print(mouse.position)
    # for i in range(4):
    #     interface.agent.act(interface.game)
    # interface.run_custom_game()
    # interface.test()
    # interface.test_loop()
    
    # new_width = 600
    # new_height = 500
    # import cv2
    # resized_image = cv2.resize(img, (256, 256))
    # cv2.imwrite("test2.png", resized_image)
    # while True:
    #     mouse = Controller()
    #     print(mouse.position)
    #     time.sleep(2)
