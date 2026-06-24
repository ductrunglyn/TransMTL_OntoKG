# conf.py (updated)
DEVICE = "cuda"

# model / lengths
LEN_IN = 512
LEN_OUT = 128
NUM_LAYER = 4
D_MODEL = 300
NUM_HEADS = 6
DFF = 1024
DROPOUT = 0.3
BATCH_SIZE = 24
NUM_WORKERS = 4
NUM_EPOCHS = 100

# learning rates (used in param-groups)
LR_BASE = 5e-5   
WEIGHT_DECAY = 1e-3
CLIP_NORM = 1.0

# label smoothing for summary cross-entropy (0.0 disables)
LABEL_SMOOTHING = 0.15

# Sử dụng từ đồng nghĩa
SIZE_VOCAB = 40000 # có thể điều chỉnh tùy theo dữ liệu và tài nguyên
USE_SYNONYM = False # False là không sử dụng

# MMoE related defaults (if you use MMoE)
USE_MMOE = None # None là tắt, True là mở
MMOE_NUM_EXPERTS = 4
MMOE_EXPERT_HIDDEN = 1024 # có thể là None
MMOE_GATE_HIDDEN = None
MMOE_DROPOUT = 0.0
MMOE_USE_RESIDUAL = True
MMOE_GATE_TEMPERATURE = 2.0
MMOE_RESIDUAL_SCALE = 1.0
WARMUP_MMOE_EPOCHS = 2
MMOE_ENTROPY_LAMBDA = 0.0

# labels for BIOES
LABELS = {"O": 0, "B": 1, "I": 2, "E": 3, "S": 4}
IGNORE_INDEX = -100    # DO NOT change to 0! (0==PAD token / O label)
PAD_IDX = 0
FREEZE_EMBEDDINGS = False
