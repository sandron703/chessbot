# Tiny, fast config used by `make smoke` to prove the pipeline runs end to end.
# It trains on a few hundred games for a few hundred iterations; the resulting
# model plays badly, which is fine -- the point is that every stage connects.

out_dir = 'out-chess-smoke'
dataset = 'chess_uci_smoke'

eval_interval = 50
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

wandb_log = False

gradient_accumulation_steps = 1
batch_size = 16
block_size = 128

n_layer = 2
n_head = 2
n_embd = 64
dropout = 0.0
bias = False

learning_rate = 1e-3
max_iters = 200
lr_decay_iters = 200
min_lr = 1e-4
warmup_iters = 20
beta2 = 0.99

compile = False
