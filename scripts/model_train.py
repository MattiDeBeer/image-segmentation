from models.model_wrappers import TrainingWrapper
from models.UNet import UNet

Trainer = TrainingWrapper(
    model_class= UNet,
    model_compilation_args = {},
    num_workers= 0
)

Trainer.train(1)









