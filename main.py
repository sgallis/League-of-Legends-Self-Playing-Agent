import argparse
import pynput
import time
import random
import logging

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from agent.agent_lightning import AgentLightning
from utils.callbacks import RolloutValidationCallback, RolloutTestCallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor_idx", type=int, default=1)
    parser.add_argument('--train', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    args.game_res = (720, 1280)
    args.client_res = (576, 1024)
    args.password = "e2h8nef87*"
    args.verbose = False
    args.shop_delay = 0.15
    args.device = "cuda"
    args.img_shape=(256, 256)
    args.action_delay = 0
    args.minions_time = 50
    # args.game_end_time = 80
    args.game_end_time = 165
    args.games_per_epoch = 3
    args.train_epochs = 100
    args.lr = 5e-5
    args.batch_size = 32
    
    args.template_threshold = 0.6
    args.r_gold = 0.01
    args.r_dead = 0.01
    args.r_level = 0.05
    args.r_pos = 0.01

    logging.basicConfig(
    level=logging.INFO,  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test/training.log", mode="w"),  # Save logs to file
        logging.StreamHandler()               # Print to console
    ]
    )
    torch.set_float32_matmul_precision('medium')
    checkpoint_cb = ModelCheckpoint(
        monitor="val_episode_return",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:03d}-{val_episode_return:.2f}"
    )

    model = AgentLightning(args)
    
    if args.train:
        trainer = L.Trainer(
            accelerator="gpu",
            max_epochs=args.train_epochs,
            gradient_clip_val=0.5,
            log_every_n_steps=1,
            callbacks=[
                checkpoint_cb,
                RolloutValidationCallback(every_n_epochs=5),
                RolloutTestCallback(n_episodes=1, compute_rewards=True)]
        )
        trainer.fit(model)
    else:
        trainer = L.Trainer(
            accelerator="gpu",
            max_epochs=args.train_epochs,
            gradient_clip_val=0.5,
            log_every_n_steps=1,
            callbacks=[
                checkpoint_cb,
                RolloutValidationCallback(every_n_epochs=1),
                RolloutTestCallback(n_episodes=1, compute_rewards=True)],
            logger=False
        )
        trainer.test(model,
                     ckpt_path="lightning_logs/version_18/checkpoints/best-epoch=004-val_episode_return=0.05.ckpt")

    # mouse = pynput.mouse.Controller()
    # while True:
    #     print(mouse.position)
    #     time.sleep(2)
