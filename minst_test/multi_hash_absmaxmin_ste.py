import argparse
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


# -----------------------------------------------------------------------
# 新增：基于 UltraSketchLLM 论文的 AbsMaxMin 与 STE 假压缩线性层
# -----------------------------------------------------------------------
class AbsMaxMinSTELinear(nn.Module):
    '''
    实现论文中提及的 Straight-Through Estimator (STE) 与多行 AbsMaxMin 草图。
    在训练时保留完整的参数，前向传播时进行假压缩（Fake Compression），梯度直接回传给完整参数。
    '''
    def __init__(self, in_features, out_features, compress=0.03125, num_rows=3, hash_seed=2, hash_bias=False):
        super(AbsMaxMinSTELinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compress = compress
        self.num_rows = num_rows # 经验性设定为 3 行 [cite: 228]

        self.original_weight_size = out_features * in_features
        self.compressed_size = max(1, int(self.original_weight_size * compress))

        # 【核心改变】：保留原始完整大小的训练参数 [cite: 293, 303]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if hash_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # 生成固定的多行哈希映射索引
        generator = torch.Generator()
        generator.manual_seed(hash_seed)
        indices = torch.randint(0, self.compressed_size, 
                                (num_rows, self.original_weight_size), 
                                generator=generator)
        self.register_buffer('hash_indices', indices)
        self._cached_eval_weight = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def fake_compress_absmaxmin(self):
        '''
        模拟论文中的 Fake Compression 流程。

        论文中的 AbsMin 更新规则是：当且仅当新权重的绝对值更小时，
        才用它替换当前 sketch state。这里使用稳定排序实现一个确定性的
        逐 bucket AbsMin 选择，不再依赖 scatter_ 在重复索引上的覆盖行为。
        '''
        weight_flat = self.weight.view(-1)
        device = self.weight.device

        # 草图状态初始化为正无穷 [cite: 209, 341]
        sketch_states = torch.full(
            (self.num_rows, self.compressed_size),
            float('inf'),
            device=device,
            dtype=weight_flat.dtype,
        )

        abs_w = torch.abs(weight_flat)

        # 先按绝对值稳定排序；相同绝对值时保留原始地址顺序，等价于论文中顺序处理权重。
        order_by_abs = torch.argsort(abs_w, stable=True)

        # 对每一行独立执行 AbsMin 更新：在每个 bucket 中保留绝对值最小的原始权重。
        for row in range(self.num_rows):
            row_indices = self.hash_indices[row]

            # 稳定按 bucket 排序，同时保持同 bucket 内按 |w| 从小到大的顺序。
            bucket_sort = torch.argsort(row_indices[order_by_abs], stable=True)
            ordered_positions = order_by_abs[bucket_sort]
            ordered_buckets = row_indices[ordered_positions]

            # 每个 bucket 的第一个元素就是该 bucket 对应的 AbsMin 结果。
            keep_mask = torch.ones_like(ordered_buckets, dtype=torch.bool)
            keep_mask[1:] = ordered_buckets[1:] != ordered_buckets[:-1]

            selected_positions = ordered_positions[keep_mask]
            selected_buckets = ordered_buckets[keep_mask]
            sketch_states[row, selected_buckets] = weight_flat[selected_positions]

        # 2. 模拟 Max 检索过程
        # 利用原始哈希索引从草图状态中提取值 [cite: 343]
        gathered = torch.gather(sketch_states, 1, self.hash_indices)
        # 跨越多行取最大值 (Max操作) 
        reconstructed_weight_flat, _ = torch.max(gathered, dim=0)
        
        # 清理未被映射到的空位（如果有的话）
        reconstructed_weight_flat[reconstructed_weight_flat == float('inf')] = 0.0
        
        return reconstructed_weight_flat.view(self.out_features, self.in_features)

    def forward(self, input):
        if self.training:
            # 训练模式：执行 Fake Compression 并使用 STE 传递梯度
            self._cached_eval_weight = None
            w_compressed = self.fake_compress_absmaxmin()
            
            # 直通估计器 (STE): 前向计算使用压缩后的权重，反向传播跳过压缩操作，直接传给 self.weight [cite: 303, 306]
            w_ste = w_compressed.detach() - self.weight.detach() + self.weight
            
            return F.linear(input, w_ste, self.bias)
        else:
            # 推理模式下缓存压缩后的权重，避免每个 batch 重复执行 fake compression。
            if self._cached_eval_weight is None or self._cached_eval_weight.device != self.weight.device:
                self._cached_eval_weight = self.fake_compress_absmaxmin().detach()
            return F.linear(input, self._cached_eval_weight, self.bias)


def get_equivalent_compression(input_dim, output_dim, nhu, nhLayers, compress):
    return compress


def build_results_path(args):
    if args.results_path is not None:
        return args.results_path

    model_name = 'absmaxmin_ste_hashed' if args.hashed else 'dense'
    return f'mnist_{model_name}_rows{args.num_rows}_compress{args.compress}_seed{args.seed}.json'


def save_results(args, parameter_count, history, test_loss, test_acc):
    results_path = build_results_path(args)
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    payload = {
        'args': vars(args),
        'parameter_count': parameter_count,
        'history': history,
        'final_test': {
            'loss': test_loss,
            'accuracy': test_acc,
        },
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print('Saved results to {}'.format(results_path))


def load_dense_initialization(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)

    loaded_layers = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, AbsMaxMinSTELinear):
            continue

        weight_key = '{}.weight'.format(module_name)
        if weight_key not in state_dict:
            raise KeyError('Missing dense weight for {}'.format(module_name))
        if state_dict[weight_key].shape != module.weight.shape:
            raise ValueError(
                'Shape mismatch for {}: dense {} vs hashed {}'.format(
                    module_name, tuple(state_dict[weight_key].shape), tuple(module.weight.shape)
                )
            )

        module.weight.data.copy_(state_dict[weight_key].to(device=device, dtype=module.weight.dtype))

        bias_key = '{}.bias'.format(module_name)
        if module.bias is not None:
            if bias_key not in state_dict:
                raise KeyError('Missing dense bias for {}'.format(module_name))
            if state_dict[bias_key].shape != module.bias.shape:
                raise ValueError(
                    'Shape mismatch for {}: dense {} vs hashed {}'.format(
                        bias_key, tuple(state_dict[bias_key].shape), tuple(module.bias.shape)
                    )
                )
            module.bias.data.copy_(state_dict[bias_key].to(device=device, dtype=module.bias.dtype))
        loaded_layers += 1

    if loaded_layers == 0:
        raise ValueError('No AbsMaxMinSTELinear layers found for dense initialization')

    print('Loaded dense initialization from {}'.format(checkpoint_path))


def save_model_checkpoint(model, path):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), path)
    print('Saved model checkpoint to {}'.format(path))


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch UltraSketchLLM AbsMaxMin STE',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--nhLayers', type=int, default=1, help='# hidden layers, excluding input/output layers')
    parser.add_argument('--nhu', type=int, default=1000, help='Number of hidden units')
    parser.add_argument('--hashed', default=False, action='store_true', help='Enable hashing')
    
    # 新增：控制多行草图的行数
    parser.add_argument('--num-rows', type=int, default=3, help='Number of rows for AbsMaxMin sketch')
    
    parser.add_argument('--compress', type=float, default=0.03125, help='Compression rate')
    parser.add_argument('--hash-bias', default=False, action='store_true', help='Hash bias terms')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate at t=0')
    parser.add_argument('--decay-factor', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--batch-size', type=int, default=50, help='Mini-batch size (1 = pure stochastic')
    parser.add_argument('--validation-percent', type=float, default=0.1, help='Percent of training data used for validation')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (SGD only)')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--l2reg', type=float, default=0.0, help='l2 regularisation')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum # of epochs')
    parser.add_argument('--patience', type=int, default=2, help='Number of epochs to wait before scaling lr.')
    parser.add_argument('--hash-seed', type=int, default=2, help='Seed for hash functions')
    parser.add_argument('--dense-init-path', type=str, default=None, help='Dense checkpoint path used to initialize hashed STE weights')
    parser.add_argument('--results-path', type=str, default=None, help='Path to save training metrics as JSON')
    parser.add_argument('--save-model-path', type=str, default='mnist.pt', help='Path to save the final model checkpoint')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    if args.num_rows < 1:
        parser.error('--num-rows must be >= 1')
    if not 0.0 < args.validation_percent < 1.0:
        parser.error('--validation-percent must be in (0, 1)')
    if args.compress <= 0.0:
        parser.error('--compress must be > 0')

    print(args)
    return args


def load_data(batch_size, validation_percent, kwargs):
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    num_train = len(train_dataset)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(math.floor(validation_percent * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, valid_loader, test_loader


def train(model, device, train_loader, optimizer, epoch, log_interval=5):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()), end='\r')
        train_loss += loss.item() * data.size(0)

    return train_loss / len(train_loader.sampler)


def evaluate(model, device, loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(loader.sampler)
    accuracy = 100. * correct / len(loader.sampler)

    return loss, accuracy


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, nhLayers=1, nhu=1000, compress=1.0, dropout=0.25):
        super(Net, self).__init__()
        self.nhLayers = nhLayers
        self.input_dim = input_dim
        c_nhu = round(nhu * compress)

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, c_nhu)
        self.dropout1 = nn.Dropout(dropout)

        for layer in range(2, nhLayers + 1):
            setattr(self, 'linear' + str(layer), nn.Linear(c_nhu, c_nhu))
            setattr(self, 'dropout' + str(layer), nn.Dropout(dropout))

        self.linear_out = nn.Linear(c_nhu, output_dim)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.dropout0(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)

        for layer in range(2, self.nhLayers + 1):
            x = F.relu(getattr(self, 'linear' + str(layer))(x))
            x = getattr(self, 'dropout' + str(layer))(x)

        x = self.linear_out(x)
        return F.log_softmax(x, dim=1)


class HashedNet(nn.Module):
    def __init__(self, input_dim, output_dim, nhLayers=1, nhu=1000,
                 compress=1.0, dropout=0.25, hash_seed=2, num_rows=3, hash_bias=False):
        super(HashedNet, self).__init__()
        self.nhLayers = nhLayers
        self.input_dim = input_dim

        self.dropout0 = nn.Dropout(dropout)
        # 将原始的线性层替换为基于 AbsMaxMin 假压缩的层
        self.linear1 = AbsMaxMinSTELinear(
            input_dim, nhu, compress, num_rows=num_rows, hash_seed=hash_seed, hash_bias=hash_bias
        )
        self.dropout1 = nn.Dropout(dropout)

        for layer in range(2, nhLayers + 1):
            setattr(
                self,
                'linear' + str(layer),
                AbsMaxMinSTELinear(
                    nhu,
                    nhu,
                    compress,
                    num_rows=num_rows,
                    hash_seed=hash_seed + layer - 1,
                    hash_bias=hash_bias,
                ),
            )
            setattr(self, 'dropout' + str(layer), nn.Dropout(dropout))

        self.linear_out = AbsMaxMinSTELinear(
            nhu,
            output_dim,
            compress,
            num_rows=num_rows,
            hash_bias=hash_bias,
            hash_seed=hash_seed + nhLayers,
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.dropout0(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)

        for layer in range(2, self.nhLayers + 1):
            x = F.relu(getattr(self, 'linear' + str(layer))(x))
            x = getattr(self, 'dropout' + str(layer))(x)

        x = self.linear_out(x)
        return F.log_softmax(x, dim=1)


def main():
    args = parse_arguments()

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    tr_loader, val_loader, test_loader = load_data(args.batch_size, args.validation_percent, kwargs)
    input_dim = 784
    output_dim = 10

    if args.hashed:
        model = HashedNet(input_dim, output_dim, args.nhLayers, args.nhu,
                          args.compress, args.dropout, args.hash_seed, args.num_rows,
                          args.hash_bias).to(device)
        if args.dense_init_path is not None:
            load_dense_initialization(model, args.dense_init_path, device)
    else:
        eq_compress = get_equivalent_compression(input_dim, output_dim, args.nhu, args.nhLayers, args.compress)
        model = Net(input_dim, output_dim, args.nhLayers, args.nhu,
                    eq_compress, args.dropout).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.l2reg)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=args.decay_factor,
                                                     patience=args.patience,
                                                     verbose=True)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The number of trainable parameters is: {}'.format(parameter_count))

    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train(model, device, tr_loader, optimizer, epoch)
        val_loss, val_acc = evaluate(model, device, val_loader)
        scheduler.step(val_loss)
        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })
        print('\nEpoch {} Train loss: {:.3f} Val loss: {:.3f} Val acc: {:.2f}%'.format(
              epoch, tr_loss, val_loss, val_acc))

    test_loss, test_acc = evaluate(model, device, test_loader)
    print('Test loss: {:.3f} Test acc: {:.2f}%'.format(test_loss, test_acc))
    save_results(args, parameter_count, history, test_loss, test_acc)

    if args.save_model:
        save_model_checkpoint(model, args.save_model_path)


if __name__ == '__main__':
    main()