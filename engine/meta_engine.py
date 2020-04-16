import torch
import torch.nn.functional as F
from utils import AverageMeter, to_var
from engine.base_engine import accuracy
import copy


# Meta-Wegiht-Net
def train_mw(train_loader, valid_loader,
             model, vnet,
             lr,
             optimizer_a, optimizer_c,
             epoch, print_freq, writer):
    """
    Train for one epoch on the training set
    @param train_loader:        imbalanced train data loader
    @param valid_loader:   meta data loader
    @param model:   classifier
    @param vnet:    weight net
    @param lr:      args.lr, initial set lr
    @param optimizer_a: 优化 classifier
    @param optimizer_c: 优化 VNet
    @param epoch:
    @param print_freq:  cmd print log
    @param writer:      tensorboard log
    @return:
    """
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    begin_step = (epoch - 1) * len(train_loader)
    meta_update_step = 10

    # total_t = 0

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        # ---------------------
        # Formulating learning manner of classifier network

        # meta_model 和 model 结构一样
        # meta_model copy 当前 model 参数，在 train data 上更新后，再在 meta data 上的更新 vnet params
        # 更新完 vnet params，再用 new weight 来实际计算 model 的更新

        if i % meta_update_step == 0:  # 每 10 个 batch 更新一次 vnet
            meta_model = copy.deepcopy(model)  # 0.02045273780822754

            # t1 = time.time()
            y_f_hat = meta_model(input_var)  # [100,10] batch_size=100

            cost = F.cross_entropy(y_f_hat, target_var, reduction='none')  # [100,1], 不用 elementwise_mean
            cost_v = torch.reshape(cost, (len(cost), 1))  # _v 表示作为 vnet 输入, bs=100 for vnet

            v_lambda = vnet(cost_v)  # [100, 1] sigmoid_()
            v_sum = torch.sum(v_lambda)  # 100 个 weights 之和
            v_lambda_norm = v_lambda / v_sum if v_sum != 0 else v_lambda

            # weighted loss, vnet 已经引入了计算图; 与 mean loss 粒度一样
            l_f_meta = torch.sum(cost_v * v_lambda_norm)  # vnet 模型的输入输出 mul
            # 并不是 loss 之和，而是加权 loss，v_lambda_norm 已经归一化; 和 mean loss 粒度一样

            # 没有直接 l_f_meta.backwards(), 这样会一并更新 vnet，因为想只更新 meta_model.params()

            # zero the model param grads
            meta_model.zero_grad()
            grads = torch.autograd.grad(  # return backward grads
                l_f_meta,  # outputs
                (meta_model.params()),  # inputs, graph leaves
                create_graph=True,
                only_inputs=True,
            )

            # 先调整 lr，再更新 meta_model 参数
            meta_lr = lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
            # meta_lr = lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 70)))
            meta_model.update_params(lr_inner=meta_lr, source_params=grads)
            del grads

            writer.add_scalar('Train/meta_lr', meta_lr, global_step=epoch)

            # ---------------------
            # Updating parameters of Meta-Weight-Net
            # vnet, meta-learned global params

            # meta data: valid_loader
            val_input, val_target = next(iter(valid_loader))  # 每次全部取出
            val_input_var = to_var(val_input, requires_grad=False)
            val_target_var = to_var(val_target, requires_grad=False)

            y_g_hat = meta_model(val_input_var)
            l_g_meta = F.cross_entropy(y_g_hat, val_target_var)

            # acc on meta data
            prec_meta = accuracy(y_g_hat.data, val_target_var.data, topk=(1,))[0]

            # 优化 vnet, fixed lr 1e-5
            optimizer_c.zero_grad()
            l_g_meta.backward()  # 反向传播，vnet 更新，train data 阶段 meta_model 保留了 grads graph
            optimizer_c.step()

            # 没更新 vnet 前，meta data 上的 CE loss 和 acc
            meta_losses.update(l_g_meta.item(), input.size(0))
            meta_top1.update(prec_meta.item(), input.size(0))

        # ---------------------
        # Updating parameters of classifier network

        y_f = model(input_var)  # [100,10]
        cost_w = F.cross_entropy(y_f, target_var, reduction='none')
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))  # [100,1] bs=100 for vnet

        # acc on train data
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        # vnet 更新后，计算 new weight
        with torch.no_grad():
            w_new = vnet(cost_v)

        v_sum = torch.sum(w_new)
        w_v = w_new / v_sum if v_sum != 0 else w_new

        # 优化 outer model
        l_f = torch.sum(cost_v * w_v)
        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()  # 更新 model 参数

        # 更新 vnet 后，在 trian data 上的 weighted CE loss[计入新的weighted] 和 acc
        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        writer.add_scalars('Train/loss', {
            'meta_loss': meta_losses.avg,
            'train_loss': losses.avg
        }, global_step=begin_step + i)

        writer.add_scalars('Train/top1_acc', {
            'meta_acc': meta_top1.avg,
            'train_acc': top1.avg
        }, global_step=begin_step + i)

        # idx in trainloader
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses, meta_loss=meta_losses, top1=top1, meta_top1=meta_top1))


# v1: softmax pre output logits as vnet input
def train_mw_v1(train_loader, valid_loader,
                model, vnet,
                lr,
                optimizer_a, optimizer_c,
                epoch, print_freq, writer):
    """
    Train for one epoch on the training set
    @param train_loader:        imbalanced train data loader
    @param valid_loader:   meta data loader
    @param model:   classifier
    @param vnet:    weight net
    @param lr:      args.lr, initial set lr
    @param optimizer_a: 优化 classifier
    @param optimizer_c: 优化 VNet
    @param epoch:
    @param print_freq:  cmd print log
    @param writer:      tensorboard log
    @return:
    """
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    begin_step = (epoch - 1) * len(train_loader)
    meta_update_step = 10

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        # ---------------------
        # Formulating learning manner of classifier network

        # meta_model 和 model 结构一样
        # meta_model copy 当前 model 参数，在 train data 上更新后，再在 meta data 上的更新 vnet params
        # 更新完 vnet params，再用 new weight 来实际计算 model 的更新

        if i % meta_update_step == 0:  # 每 10 个 batch 更新一次 vnet
            meta_model = copy.deepcopy(model)  # 0.02045273780822754

            y_f_hat = meta_model(input_var)  # [100,10] batch_size=100

            cost = F.cross_entropy(y_f_hat, target_var, reduction='none')  # [100,1], 不用 elementwise_mean
            cost_v = torch.reshape(cost, (len(cost), 1))  # bs=100 for vnet

            v_lambda = vnet(y_f_hat)  # [100, 1] sigmoid_()
            v_sum = torch.sum(v_lambda)  # 100 个 weights 之和
            v_lambda_norm = v_lambda / v_sum if v_sum != 0 else v_lambda

            # weighted loss, vnet 已经引入了计算图; 与 mean loss 粒度一样
            l_f_meta = torch.sum(cost_v * v_lambda_norm)  # vnet 模型的输入输出 mul
            # 并不是 loss 之和，而是加权 loss，v_lambda_norm 已经归一化; 和 mean loss 粒度一样

            # 没有直接 l_f_meta.backwards(), 这样会一并更新 vnet，因为想只更新 meta_model.params()

            # zero the model param grads
            meta_model.zero_grad()
            grads = torch.autograd.grad(  # return backward grads
                l_f_meta,  # outputs
                (meta_model.params()),  # inputs, graph leaves
                create_graph=True,
                only_inputs=True,
            )

            # 先调整 lr，再更新 meta_model 参数
            meta_lr = lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
            # meta_lr = lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 70)))
            meta_model.update_params(lr_inner=meta_lr, source_params=grads)
            del grads

            writer.add_scalar('Train/meta_lr', meta_lr, global_step=epoch)

            # ---------------------
            # Updating parameters of Meta-Weight-Net
            # vnet, meta-learned global params

            # meta data: valid_loader
            val_input, val_target = next(iter(valid_loader))  # 每次全部取出
            val_input_var = to_var(val_input, requires_grad=False)
            val_target_var = to_var(val_target, requires_grad=False)

            y_g_hat = meta_model(val_input_var)
            l_g_meta = F.cross_entropy(y_g_hat, val_target_var)

            # acc on meta data
            prec_meta = accuracy(y_g_hat.data, val_target_var.data, topk=(1,))[0]

            # 优化 vnet, fixed lr 1e-5
            optimizer_c.zero_grad()
            l_g_meta.backward()  # 反向传播，vnet 更新，train data 阶段 meta_model 保留了 grads graph
            optimizer_c.step()

            # 没更新 vnet 前，meta data 上的 CE loss 和 acc
            meta_losses.update(l_g_meta.item(), input.size(0))
            meta_top1.update(prec_meta.item(), input.size(0))

        # ---------------------
        # Updating parameters of classifier network

        y_f = model(input_var)  # [100,10]
        cost_w = F.cross_entropy(y_f, target_var, reduction='none')
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))  # [100,1] bs=100 for vnet

        # acc on train data
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        # vnet 更新后，计算 new weight
        with torch.no_grad():
            w_new = vnet(y_f)

        v_sum = torch.sum(w_new)
        w_v = w_new / v_sum if v_sum != 0 else w_new

        # 优化 outer model
        l_f = torch.sum(cost_v * w_v)
        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()  # 更新 model 参数

        # 更新 vnet 后，在 trian data 上的 weighted CE loss[计入新的weighted] 和 acc
        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        writer.add_scalars('Train/loss', {
            'meta_loss': meta_losses.avg,
            'train_loss': losses.avg
        }, global_step=begin_step + i)

        writer.add_scalars('Train/top1_acc', {
            'meta_acc': meta_top1.avg,
            'train_acc': top1.avg
        }, global_step=begin_step + i)

        # idx in trainloader
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses, meta_loss=meta_losses, top1=top1, meta_top1=meta_top1))


# v2: batch_loss -> batch_weight
def train_mw_v2(train_loader, valid_loader,
                model, vnet,
                lr,
                optimizer_a, optimizer_c,
                epoch, print_freq, writer):
    """
    Train for one epoch on the training set
    @param train_loader: imbalanced train data loader
    @param valid_loader: meta data loader
    @param model:   classifier
    @param vnet:    weight net
    @param lr:      args.lr, initial set lr
    @param optimizer_a: 优化 classifier
    @param optimizer_c: 优化 VNet
    @param epoch:
    @param print_freq:  cmd print log
    @param writer:      tensorboard log
    @return:
    """
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    begin_step = (epoch - 1) * len(train_loader)
    meta_update_step = 10

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        # ---------------------
        # Formulating learning manner of classifier network
        if i % meta_update_step == 0:  # 每 10 个 batch 更新一次 vnet
            meta_model = copy.deepcopy(model)

            # t1 = time.time()
            y_f_hat = meta_model(input_var)  # [100,10] batch_size=100
            # 考虑用此作为 vnet 输入?

            cost = F.cross_entropy(y_f_hat, target_var, reduction='none')  # [100,1], 不用 elementwise_mean
            # cost_v = torch.reshape(cost, (len(cost), 1))  # _v 表示作为 vnet 输入
            cost_v = torch.reshape(cost, (1, len(cost)))  # [1,100] bs=1 for v2

            v_lambda = vnet(cost_v)  # [1,100] sigmoid_()
            v_sum = torch.sum(v_lambda)  # 100 个 weights 之和
            v_lambda_norm = v_lambda / v_sum if v_sum != 0 else v_lambda

            # weighted loss, vnet 已经引入了计算图; 与 mean loss 粒度一样
            l_f_meta = torch.sum(cost_v * v_lambda_norm)  # vnet 模型的输入输出 mul
            # 并不是 loss 之和，而是加权 loss，v_lambda_norm 已经归一化; 和 mean loss 粒度一样

            # 没有直接 l_f_meta.backwards(), 这样会一并更新 vnet，因为想只更新 meta_model.params()

            # zero the model param grads
            meta_model.zero_grad()
            grads = torch.autograd.grad(  # return backward grads
                l_f_meta,  # outputs
                (meta_model.params()),  # inputs, graph leaves
                create_graph=True,
                only_inputs=True,
            )

            # 先调整 lr，再更新 meta_model 参数
            meta_lr = lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
            # meta_lr = lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 70)))
            meta_model.update_params(lr_inner=meta_lr, source_params=grads)
            del grads

            writer.add_scalar('Train/meta_lr', meta_lr, global_step=epoch)

            # ---------------------
            # Updating parameters of Meta-Weight-Net
            # vnet, meta-learned global params

            # meta data: valid_loader
            val_input, val_target = next(iter(valid_loader))  # 每次全部取出
            val_input_var = to_var(val_input, requires_grad=False)
            val_target_var = to_var(val_target, requires_grad=False)

            y_g_hat = meta_model(val_input_var)
            l_g_meta = F.cross_entropy(y_g_hat, val_target_var)

            # acc on meta data
            prec_meta = accuracy(y_g_hat.data, val_target_var.data, topk=(1,))[0]

            # 优化 vnet, fixed lr 1e-5
            optimizer_c.zero_grad()
            l_g_meta.backward()  # 反向传播，vnet 更新，train data 阶段 meta_model 保留了 grads graph
            optimizer_c.step()

            # 没更新 vnet 前，meta data 上的 CE loss 和 acc
            meta_losses.update(l_g_meta.item(), input.size(0))
            meta_top1.update(prec_meta.item(), input.size(0))

        # ---------------------
        # Updating parameters of classifier network

        y_f = model(input_var)  # [100,10]
        cost_w = F.cross_entropy(y_f, target_var, reduction='none')
        cost_v = torch.reshape(cost_w, (1, len(cost_w)))  # [1,100] bs=1 for v2

        # acc on train data
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        # vnet 更新后，计算 new weight
        with torch.no_grad():
            w_new = vnet(cost_v)

        v_sum = torch.sum(w_new)
        w_v = w_new / v_sum if v_sum != 0 else w_new

        # 优化 outer model
        l_f = torch.sum(cost_v * w_v)
        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()  # 更新 model 参数

        # 更新 vnet 后，在 trian data 上的 weighted CE loss[计入新的weighted] 和 acc
        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        writer.add_scalars('Train/loss', {
            'meta_loss': meta_losses.avg,
            'train_loss': losses.avg
        }, global_step=begin_step + i)

        writer.add_scalars('Train/top1_acc', {
            'meta_acc': meta_top1.avg,
            'train_acc': top1.avg
        }, global_step=begin_step + i)

        # idx in trainloader
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses, meta_loss=meta_losses, top1=top1, meta_top1=meta_top1))


# Base model
def train_base(train_loader, model,
               criterion, optimizer_a,
               epoch, print_freq, writer):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    begin_step = (epoch - 1) * len(train_loader)
    # total_t = 0

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        # t1 = time.time()

        output = model(input_var)
        loss = criterion(output, target_var)  # reduction='mean', [1,]
        prec_train = accuracy(output.data, target_var.data, topk=(1,))[0]

        # 普通更新
        optimizer_a.zero_grad()
        loss.backward()
        optimizer_a.step()

        # CE loss, reduction='mean'
        losses.update(loss.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        writer.add_scalar('Train/loss', losses.avg, global_step=begin_step + i)
        writer.add_scalar('Train/top1_acc', top1.avg, global_step=begin_step + i)

        # total_t += time.time() - t1

        # idx in trainloader
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1))

            # print('each iteration time:', total_t / print_freq)
            # total_t = 0
