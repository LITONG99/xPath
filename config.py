import os
import logging


HGN_TYPE = 'simplehgn'
DATASET = 'acm'
N_LAYER = 2
REPEAT_ID = 1 # experiment id
GPU = 0

abs_path = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(abs_path, 'log')
os.makedirs(log_dir, exist_ok=True)
data_dir = os.path.join(abs_path, 'data', DATASET)
result_dir = os.path.join(abs_path, 'results', HGN_TYPE)
os.makedirs(result_dir, exist_ok=True)
ckpt_dir = os.path.join(abs_path, 'ckpt', DATASET)
os.makedirs(ckpt_dir, exist_ok=True)


if DATASET == 'acm':
    XPATH_BEAM = 5
    XPATH_SAMPLE_N = 5
    XPATH_TOP_K = 4
elif DATASET == 'dblp' and HGN_TYPE == 'simplehgn' and N_LAYER == 2:
    XPATH_BEAM = 10
    XPATH_SAMPLE_N = 10
    XPATH_TOP_K = 3
else:
    XPATH_BEAM = 2
    XPATH_SAMPLE_N = 10
    XPATH_TOP_K = 4


pred_list_path = f"{ckpt_dir}/{HGN_TYPE}_{DATASET}_pred_list_{N_LAYER}.json"
#hgn_path = f"{ckpt_dir}/{HGN_TYPE}_{DATASET}_{N_LAYER}"
# To use trained model:
hgn_path = f"{ckpt_dir}/bk/{HGN_TYPE}_{DATASET}_{N_LAYER}"
graph_path = f"{data_dir}/{DATASET}_graph.bin"
result_path = f'{result_dir}/{DATASET}_l{N_LAYER}_xpath2s_{XPATH_BEAM}_{XPATH_SAMPLE_N}_exp{REPEAT_ID}'

if HGN_TYPE=='simplehgn':
    index_path = f"{data_dir}/{DATASET}_index_60.bin"
else:
    index_path = f"{data_dir}/{DATASET}_index_2000.bin"

if DATASET == 'dblp':
    TARGET_NTYPE = 'author'
    NUM_CLASSES = 4

elif DATASET == 'acm':
    TARGET_NTYPE = 'paper'
    NUM_CLASSES = 3

elif DATASET == 'imdb':
    TARGET_NTYPE = 'movie'
    NUM_CLASSES = 3

# init logger
log_root = log_dir + f'/{HGN_TYPE}'
os.makedirs(log_root, exist_ok=True)
log_file = log_root + f'/{DATASET}_l{N_LAYER}_r{REPEAT_ID}.log'
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()
fmt = '%(asctime)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(fmt)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger = logging.getLogger('updateSecurity')
logger.setLevel('DEBUG')
logger.addHandler(file_handler)
logger.addHandler(console_handler)