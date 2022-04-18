
common_config = {
    'data_dir': '../data/labels',
    'img_width': 384,
    'img_height': 128,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
}

train_config = {
    'epochs': 100,
    'train_batch_size': 64,
    'valid_batch_size': 64,
    'lr': 0.0002,
    'show_interval': 1,
    'valid_epoch': 1,
    'save_epoch': 1,
    'cpu_workers': 8,
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoints/'
}
train_config.update(common_config)

evaluate_config = {
    'eval_epoch': 1,
    'eval_batch_size': 64,
    'cpu_workers': 8,
    'decode_method': 'greedy',
    # 'decode_method': 'beam_search',
    'beam_size': 10,
}
evaluate_config.update(common_config)
