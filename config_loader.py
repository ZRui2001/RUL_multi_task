"""
从配置文件config.yaml加载全局变量
"""

import yaml
import torch


def load_common_config():
    # 加载配置文件
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    models_config = config['models']
    seeds = config['seeds']
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    return models_config, seeds, device


def load_data_config(dataset):
    # 加载配置文件
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # 提取配置项
    data_config = config['data'][dataset]

    # 数据相关配置
    val_bat = data_config['val_bat']
    test_bat = data_config['test_bat']
    bats = data_config['bats']
    lookback_len = data_config['lookback_len']
    pred_len = data_config['pred_len']
    rated_capacity = data_config['rated_capacity']
    batch_size = data_config['batch_size']

    return data_config, val_bat, test_bat, bats, lookback_len, pred_len, rated_capacity, batch_size
