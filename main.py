import argparse
import pynput
import time
import random
import logging

import torch
from torchvision.models import resnet18, ResNet18_Weights
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from agent.agent_lightning import AgentLightning
from utils.callbacks import RolloutValidationCallback, RolloutTestCallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    args.game_res = (720, 1280)
    args.client_res = (576, 1024)
    args.password = "e2h8nef87*"
    args.verbose = False
    args.shop_delay = 0.15
    # args.device = "cuda"
    args.img_shape=(224, 224)
    args.action_delay = 0
    args.minions_time = 50
    # args.game_end_time = 30
    args.game_end_time = 300
    args.games_per_epoch = 8
    args.train_epochs = 100
    args.lr = 3e-4
    args.batch_size = 256
    
    args.m_pos = (20, 35)

    args.template_threshold = 0.6
    args.template_path = "assets/MissFortune_map.png"
    args.r_gold = 0.01
    args.r_dead = 0.01
    args.r_level = 0.01
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
        monitor="episode_return",
        mode="max",
        save_top_k=5,
        save_last=True,
        filename="best-{epoch:03d}-{episode_return:.2f}"
    )

    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = AgentLightning(backbone, args)

    from utils.visualize import view_template_matching
    view_template_matching(model.game, "assets/MissFortune_map.png")

    if args.train:
        trainer = L.Trainer(
            accelerator="gpu",
            max_epochs=args.train_epochs,
            gradient_clip_val=0.5,
            log_every_n_steps=1,
            callbacks=[
                checkpoint_cb,
                # RolloutValidationCallback(every_n_epochs=5),
                RolloutTestCallback(n_episodes=1, compute_rewards=True)]
        )
        trainer.fit(
            model,
            # ckpt_path="last"
        )
    else:
        trainer = L.Trainer(
            accelerator="gpu",
            max_epochs=args.train_epochs,
            gradient_clip_val=0.5,
            log_every_n_steps=1,
            callbacks=[
                checkpoint_cb,
                RolloutValidationCallback(every_n_epochs=1),
                RolloutTestCallback(
                    n_episodes=1,
                    compute_rewards=True,
                    sample=True,
                    )],
            logger=False
        )
        trainer.test(
            model,
            ckpt_path="lightning_logs/version_14/checkpoints/best-epoch=032-episode_return=-1.25.ckpt",
            )

    # mouse = pynput.mouse.Controller()
    # while True:
    #     print(mouse.position)
    #     time.sleep(2)
