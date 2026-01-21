import argparse
import pynput
import time
import random
import logging

import torch
import lightning as L

from agent.agent_lightning import AgentLightning

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor_idx", type=int, default=1)
    args = parser.parse_args()
    args.game_res = (720, 1280)
    args.client_res = (576, 1024)
    args.password = "e2h8nef87*"
    args.verbose = False
    args.game_end_time = 120
    args.shop_delay = 0.15
    args.device = "cuda"
    args.img_shape=(256, 256)
    args.action_delay = 0.15
    args.minions_time = 48
    args.games_per_epoch = 3
    args.train_epochs = 10
    args.lr = 1e-3
    args.batch_size = 8

    logging.basicConfig(
    level=logging.INFO,  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log", mode="w"),  # Save logs to file
        logging.StreamHandler()               # Print to console
    ]
    )
    torch.set_float32_matmul_precision('medium')
    model = AgentLightning(args)
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=args.train_epochs,
        gradient_clip_val=0.5,
        log_every_n_steps=1,
    )

    trainer.fit(model)

    # mouse = pynput.mouse.Controller()
    # while True:
    #     print(mouse.position)
    #     time.sleep(2)
