# nanoGPT config -- the prototype model. ~3.2M params.
#
#   cd ../nanoGPT && python train.py "../prototype./configs/train_chess_small.py"
#
# Budget check: 12 * n_layer * n_embd^2 = 3.15M params, so ~64M training tokens
# (~210k self-play games at ~310 tokens/game) is the Chinchilla-ish target.
# block_size 384 covers a ~190-ply game, which is most of them; longer games
# train as mid-game windows, which is exactly what inference does once a game
# outgrows the context.

out_dir = 'out-chess-small'
dataset = 'chess_uci'

eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = False   # only save when val loss improves

wandb_log = False
wandb_project = 'chess-nanogpt'
wandb_run_name = 'small'

gradient_accumulation_steps = 1
batch_size = 64
block_size = 384                 # ~24.6k tokens per iteration

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.05                   # a little, since we re-epoch the self-play set
bias = False

learning_rate = 1e-3
max_iters = 12000
lr_decay_iters = 12000
min_lr = 1e-4
warmup_iters = 200
beta2 = 0.99                     # few tokens per iter -> slower second moment
grad_clip = 1.0

compile = True                   # set False if torch.compile errors on your setup
