from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, ExponentialLR
import statistics
import torch.nn.functional as F
import torch
from tqdm import tqdm
import datetime
import os
from MahJong_CNN_model6_largeV2 import MahJongCNNNet6_LargeV2


validation_acc = 0.0
validation_loss = 100.0

torch.set_num_threads(2)


def prepare_device(n_gpu_use, main_id=0):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    main_id for specify main gpu
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:%d" % main_id if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def workload(network, DS, loader, epoch_it, save_dir, device, is_training=True):
    """
    iteration: iteration in epoch
    run_total: total runs to finish an epoch
    split_ratio: train/valid split ratio
    vDS: validation dataset
    vLdr: validation loader
    """
    global validation_acc
    global validation_loss

    if is_training:
        network.train()
    else:
        network.eval()

    correct = 0
    loss_list = []
    for i, d in enumerate(
        tqdm(loader, leave=False, desc="Training: " if is_training else "Validating: ")
    ):

        obs = d[0].to(device).float()
        mask = d[1].to(device)
        act = d[2].to(device).long()
        input = {"observation": obs, "action_mask": mask}

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            logits = network(input)
            loss = F.cross_entropy(logits, act)
            loss_list.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), max_norm=1.0, norm_type=2
            )
            optimizer.step()
            warmup_scheduler.step()
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, d[2].to(device)).sum().item()
            # log step value to tensorboard
            # writer.add_scalar("loss_step", loss.item(), epoch_it * len(loader) + i)

        else:
            with torch.no_grad():
                logits = network(input)
                loss = F.cross_entropy(logits, act)
                loss_list.append(loss.item())
                pred = logits.argmax(dim=1)
                correct += torch.eq(pred, d[2].to(device)).sum().item()

    if is_training:
        acc = correct / len(DS)
        torch.save(
            network.state_dict(),
            save_dir + "/checkpoint/%d.pkl" % epoch_it,
        )
        writer.add_scalar(
            "loss_global/loss_train", statistics.mean(loss_list), epoch_it
        )
        writer.add_scalar("acc_global/acc_train", acc, epoch_it)
    else:
        acc = correct / len(DS)
        writer.add_scalar(
            "loss_global/loss_validate", statistics.mean(loss_list), epoch_it
        )
        writer.add_scalar("acc_global/acc_validate", acc, epoch_it)
        if acc > validation_acc:
            torch.save(network.state_dict(), save_dir + "/checkpoint/best_v_acc.pkl")
            validation_acc = acc


if __name__ == "__main__":
    # configs
    model = MahJongCNNNet6_LargeV2
    data_path = "/mnt/storage/llf/hzddBot/sl_data_featurefull2"
    meta_path = "/mnt/storage/llf/hzddBot/count_featurefull2.json"

    logdir_base = "logs"
    log_note = "hzddBot_featurefull2"
    now = datetime.datetime.now()
    logdir = os.path.join(logdir_base, now.strftime("%m_%d_%H_%M") + log_note)
    sd_data_name = "/checkpoint/best_v_loss.pkl"
    sd_data_path = logdir + sd_data_name
    load_data = False  # if resume from previous training session
    writer = SummaryWriter(logdir)

    # prepare log dir
    if not os.path.exists(logdir + "/checkpoint"):
        os.makedirs(logdir + "/checkpoint")

    splitRatio = 0.93
    batchSize = 512
    epoch_total = 60
    n_gpu = 1
    main_gpu_id = 2
    lr_rate = 1e-5
    number_warmup_steps = 500

    device, device_ids = prepare_device(n_gpu, main_gpu_id)

    # Load dataset
    vDS = MahjongGBDataset(data_path, meta_path, splitRatio, 1)
    vloader = DataLoader(
        dataset=vDS, batch_size=batchSize, shuffle=False, num_workers=8
    )
    tDS = MahjongGBDataset(data_path, meta_path, 0, splitRatio)
    tloader = DataLoader(dataset=tDS, batch_size=batchSize, shuffle=True, num_workers=8)
    # Load model
    nn = model(device).to(device)

    # nn = model.CNNModel().to("cuda")
    def warmup(current_step: int):
        if current_step > number_warmup_steps:
            return 1
        else:
            return 1 / (1.003 ** (float(number_warmup_steps - current_step)))

    optimizer = torch.optim.Adam(nn.parameters(), lr=lr_rate)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)
    if load_data:
        nn.load_state_dict(torch.load(sd_data_path, map_location=torch.device(device)))
    # Train and validate
    for i in tqdm(range(epoch_total), desc="Epoch Progress: "):
        # training
        workload(
            nn,
            tDS,
            tloader,
            i,
            logdir,
            device,
            True,
        )
        # validation
        workload(nn, vDS, vloader, i, logdir, device, False)
        writer.flush()

    writer.close()
