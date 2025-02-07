import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np

from model_foundation_local_rev2 import Foundation as Foundation_local
from dataloader import TinyTomDataset
from loss_functions import foundation_loss

from utils import cosine_scheduler

if __name__ == "__main__":  
    # Hyperparameters 
    IMG_SIZE = 128
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    LEARNING_RATE_END = 0.000001
    WARMUP_LR_START = LEARNING_RATE_END
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 0
    NUM_EPOCHS = 25
    NUM_WORKERS = 16
    PIN_MEMORY = False
    SAVE_MODELS = True
    MODEL_NAME = "phileo-precursor_v09"

    # Initialise seeds    
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)

    from torch.multiprocessing import set_start_method
    set_start_method("spawn")

    augment = transforms.Compose(
        [
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    ) 

    augment_drops = transforms.Compose([
        transforms.RandomErasing(p=1.0, scale=(0.25, 0.50), ratio=(0.3, 3.3), value="random", inplace=False),
    ]) 

    train_set = TinyTomDataset(
        "/home/phimultigpu/tinyTOM/dataset/",
        "/home/phimultigpu/phileo-foundation/data_static",
        patch_size=IMG_SIZE,
        read_static_to_ram=True, # Currently does nothing    
        device="cuda",
        transform=augment,
    )    

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)  

    # 128            
    model = Foundation_local(
        input_dim=10, # B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 
        # input_dim=13, # B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12             
        depths=[3, 3, 4, 4, 5],  # 128, 64, 32, 16, 8
        dims=[32, 32, 64, 64, 128],
        img_size=128,
        latent_dim=1024,
        # dropout=[0.85, 0.90, 0.90, 0.95, 0.95], 
        dropout=None,
        activation=nn.GELU(),
    )      

    model.cuda() 
    model = torch.nn.DataParallel(model)

    weights = torch.load("/home/phimultigpu/phileo-foundation/models/phileo-precursor_v08_e025.pt")         
    model.load_state_dict(dict(weights), strict=False) 

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    lr_schedule_values = cosine_scheduler(
        LEARNING_RATE, LEARNING_RATE_END, NUM_EPOCHS + WARMUP_EPOCHS, WARMUP_EPOCHS, WARMUP_LR_START,
    )

    for epoch in range(NUM_EPOCHS + WARMUP_EPOCHS): 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule_values[epoch]

        loss_accum = {
            "loss": 0.0,
            "rec": 0.0,
            "xy": 0.0,
            "cl": 0.0,
            "b": 0.0,
            "lc": 0.0,
            "sim": 0.0,
        }

        train_pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS + WARMUP_EPOCHS}")
        for i, data in enumerate(train_pbar):
            model.train()
            inputs, labels = data
            inputs = torch.cat([inputs, torch.ones_like(inputs[:, :1, :, :])], dim=1)
            inputs = inputs.cuda()

            inputs_augs_1 = augment(inputs)
            inputs_aug_1, inputs_aug_1_mask = inputs_augs_1[:, :-1, :, :], inputs_augs_1[:, -1:, :, :]
            inputs_augs_2 = augment(inputs)
            inputs_aug_2, inputs_aug_2_mask = inputs_augs_2[:, :-1, :, :], inputs_augs_2[:, -1:, :, :]

            inputs_drop1 = augment_drops(inputs_aug_1)
            inputs_drop2 = augment_drops(inputs_aug_2)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                og_recon, og_emb, _og_emb_cnn, _og_decode, og_preds = model(inputs_drop1)
                aug_recon, aug_emb, _aug_emb_cnn, _aug_decode, aug_preds = model(inputs_drop2)

                loss, log = foundation_loss(og_recon * inputs_aug_1_mask, og_emb, og_preds, aug_recon * inputs_aug_2_mask, aug_emb, aug_preds, inputs_aug_1, inputs_aug_2, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_accum["loss"] += log["loss"]
            loss_accum["rec"] += log["rec"]
            loss_accum["xy"] += log["xy"]
            loss_accum["cl"] += log["cl"]
            loss_accum["b"] += log["b"]
            loss_accum["lc"] += log["lc"]
            loss_accum["sim"] += log["sim"]

            loss_dict = {
                "loss": f"{loss_accum['loss'] / (i + 1):0.4f}".rjust(2, ' '),
                "rec": f"{loss_accum['rec'] / (i + 1):0.4f}".rjust(2, ' '),
                "xy": f"{loss_accum['xy'] / (i + 1):0.4f}".rjust(2, ' '),
                "cl": f"{loss_accum['cl'] / (i + 1):0.4f}".rjust(2, ' '),
                "b": f"{loss_accum['b'] / (i + 1):0.4f}".rjust(2, ' '),
                "lc": f"{loss_accum['lc'] / (i + 1):0.4f}".rjust(2, ' '),
                "sim": f"{loss_accum['sim'] / (i + 1):0.4f}".rjust(2, ' '),
            }

            train_pbar.set_postfix(loss_dict)
            prepend_epoch = str(epoch + 1).rjust(3, '0')

            if i % 100 == 0:
                loss_dict["epoch"] = prepend_epoch
                loss_dict["iter"] = i

                with open(f"/home/phimultigpu/phileo-foundation/logs/{MODEL_NAME}.txt", "a") as file:
                    file.write(f"{loss_dict}\n")

        if SAVE_MODELS:
            torch.save(model.state_dict(), f"/home/phimultigpu/phileo-foundation/models/{MODEL_NAME}_e{prepend_epoch}.pt")

    print('Finished Training')
