# nanoGPT config -- the scale-up, ~25M params (the size at which published
# PGN-trained GPTs reach roughly 1300-1500 Elo).
#
#   cd ../nanoGPT && python train.py "../prototype./configs/train_chess_medium.py"
#
# This wants ~500M tokens (~1.6M self-play games), which is many hours of
# Stockfish time -- generate data with the small model already training.
# On a 6GB card the batch has to be split across grad-accum steps.

out_dir = 'out-chess-medium'
dataset = 'chess_uci'

eval_interval = 500
eval_iters = 200
log_interval = 20
always_save_checkpoint = False

wandb_log = False
wandb_project = 'chess-nanogpt'
wandb_run_name = 'medium'

gradient_accumulation_steps = 4
batch_size = 24
block_size = 384

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

learning_rate = 6e-4
# 504M tokens / (24*4*384 per iter) = ~13.7k iters per epoch; 40k is ~3 epochs.
max_iters = 40000
lr_decay_iters = 40000
min_lr = 6e-5
warmup_iters = 1000
beta2 = 0.95
grad_clip = 1.0

compile = True
