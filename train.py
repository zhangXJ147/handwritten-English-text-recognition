import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 设置环境变量
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from functools import partial
from src.dataset import IAMDataset, IAM_collate_fn
from src.evaluate import evaluate
from src.config import train_config, evaluate_config
from src.CTN import CTN
import torchvision.transforms as transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_batch(model, data, optimizer, criterion, device):
    model.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = model(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    epochs = train_config['epochs']
    train_batch_size = train_config['train_batch_size']
    eval_batch_size = evaluate_config['eval_batch_size']
    lr = train_config['lr']
    show_interval = train_config['show_interval']
    eval_epoch = evaluate_config['eval_epoch']
    save_epoch = train_config['save_epoch']
    cpu_workers = train_config['cpu_workers']

    img_width = train_config['img_width']
    img_height = train_config['img_height']
    data_dir = train_config['data_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_transform = transforms.RandomChoice([
        transforms.RandomAffine(degrees=(-2, 2), translate=(0, 0), scale=(0.9, 1),
                                shear=5, resample=False, fillcolor=255),
        transforms.RandomAffine(degrees=(-3, 3), translate=(0, 0.2), scale=(0.9, 1),
                                shear=5, resample=False, fillcolor=255),
    ])

    train_dataset = IAMDataset(root_dir=data_dir, mode='train', transform=train_transform,
                               img_height=img_height, img_width=img_width)
    eval_dataset = IAMDataset(root_dir=data_dir, mode='dev',
                              img_height=img_height, img_width=img_width)

    LEN_TRAIN_SET = train_dataset.__len__()
    LEN_VALI_SET = eval_dataset.__len__()
    print("len(train_set) =", LEN_TRAIN_SET)
    print("len(valid_set) =", LEN_VALI_SET)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=IAM_collate_fn)
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=IAM_collate_fn)

    num_class = len(IAMDataset.LABEL2CHAR) + 1
    model = CTN(
        height_patch_size=7, width_patch_size=1, embed_dims=[64, 128, 320, 512], num_classes=num_class,
        num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True)

    model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    assert save_epoch % eval_epoch == 0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for j, train_data in enumerate(train_loader):
            loss = train_batch(model, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if (j + 1) % show_interval == 0:
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                    epoch, epochs, j + 1, len(train_loader), loss / train_size))

        if i % eval_epoch == 0:
            evaluation = evaluate(model, eval_loader, criterion,
                                  decode_method=evaluate_config['decode_method'],
                                  beam_size=evaluate_config['beam_size'], len_set=LEN_VALI_SET)
            print('valid: loss={loss}, acc={acc}, avg_CER={avg_CER}, avg_WER={avg_WER}'.format(**evaluation))

        if i % save_epoch == 0:
            prefix = 'CTN0.0002'
            loss = evaluation['loss']
            avg_CER = evaluation['avg_CER']
            avg_WER = evaluation['avg_WER']
            acc = evaluation['acc']
            save_model_path = os.path.join(BASE_DIR, '..', train_config['checkpoints_dir'],
                                           f'{i}_{prefix}_loss{loss}_acc{acc}_avgCER{avg_CER}_avgWER{avg_WER}.pth')
            torch.save(model.state_dict(), save_model_path)
            print('save model at ', save_model_path)

        i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)


if __name__ == '__main__':
    main()
