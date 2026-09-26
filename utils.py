import numpy as np
import torch


class EarlyStopping_Acc:
    def __init__(
        self,
        patience=7,
        verbose=True,
        delta=0.0,
        model_name="checkpoint.pt",
        save_model=True,
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.model_name = model_name
        self.save_model = save_model
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.inf

    def __call__(self, val_acc, model=None):
        if self.best_score is None or val_acc >= self.best_score + self.delta:
            self.best_score = val_acc
            self.counter = 0
            self.val_acc_max = val_acc
            if self.save_model and model is not None:
                torch.save(model.state_dict(), self.model_name)
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
