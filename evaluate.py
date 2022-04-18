import os
import torch
from tqdm import tqdm
from src.ctc_decoder import ctc_decode
from src.utils import CER, WER
from src.dataset import IAMDataset, IAM_collate_fn
from src.CTN import CTN
from functools import partial
import torch.nn as nn
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from src.config import evaluate_config as config

torch.backends.cudnn.enabled = False

CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-\',.*"+?/'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}


def evaluate(model, dataloader, criterion, len_set,
             max_iter=None, decode_method='beam_search', beam_size=10):
    model.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []
    avg_CER = 0
    avg_WER = 0

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data]

            logits = model(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                avg_CER += CER(real, pred)
                avg_WER += WER(real, pred)
                if pred == real:
                    tot_correct += 1
                else:
                    # print(pred, real, CER(real, pred))
                    wrong_cases.append((real, pred))
                    # decode_real = [LABEL2CHAR[l] for l in real]
                    # real_string = ''.join(decode_real)
                    # decode_pred = [LABEL2CHAR[l] for l in pred]
                    # pred_string = ''.join(decode_pred)
                    # f4.writelines(pred_string + ' ')
                    # f4.writelines(real_string + '\n')
            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases,
        'avg_CER': avg_CER / len_set,
        'avg_WER': avg_WER / len_set,
    }
    # f4.close()
    return evaluation


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    data_dir = config['data_dir']
    img_width = config['img_width']
    img_height = config['img_height']
    eval_batch_size = config['eval_batch_size']
    cpu_workers = config['cpu_workers']

    eval_dataset = IAMDataset(root_dir=data_dir, mode='test',
                              img_height=img_height, img_width=img_width)
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=IAM_collate_fn)

    LEN_VALI_SET = eval_dataset.__len__()
    print("len(test_set) =", LEN_VALI_SET)

    num_class = len(IAMDataset.LABEL2CHAR) + 1
    model = CTN(
        height_patch_size=7, width_patch_size=1, embed_dims=[64, 128, 320, 512], num_classes=num_class,
        num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True)

    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    # dir_path = os.path.join(BASE_DIR, '..', 'checkpoints')
    # for filename in os.listdir(dir_path):
    # f4 = open('wrong.txt', 'w')

    pretrained_dict_path = os.path.join(BASE_DIR, '..', 'checkpoints',
        '98_CTN0.0002_loss0.9201308958159314_acc0.86869164401371_avgCER0.036691087183050246_avgWER0.13130835598629004.pth')
    pretrained_model_dict = torch.load(pretrained_dict_path)
    model.load_state_dict(pretrained_model_dict)
    model.to(device)

    evaluation = evaluate(model, eval_loader, criterion,
                          decode_method=config['decode_method'],
                          beam_size=config['beam_size'], len_set=LEN_VALI_SET)
    # print(filename)
    print('test: loss={loss}, acc={acc}, avg_CER={avg_CER}, avg_WER={avg_WER}'.format(**evaluation))
    # print('wrong_cases={wrong_cases}'.format(**evaluation))
