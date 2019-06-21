import pickle as pkl
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import argparse

def main():
	parser = argparse.ArgumentParser()
    parser.add_argument('--envname', type=str, default='Humanoid-v2')
    parser.add_argument('--rollout_data', type=str, default="expert_data/Humanoid-v2.pkl")
    args = parser.parse_args()

    