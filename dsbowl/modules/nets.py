from tqdm.autonotebook import tqdm
from datetime import timedelta
import time
import math
import torch
import numpy as np
from numbers import Number
import modules.annealings as an
from modules.preds import predict_all, predict_TTA_all


class Net:
    def __init__(self, model, optim, loss, metrics, models_dir):
        self.model = model
        self.optim = optim
        self.loss = loss
        self.metrics = metrics
        self.models_dir = models_dir

    def __call__(self, input):
        return self.model(input)

    def fit(
            self, dls, num_epochs, save_name, device, state_dict=None,
            scheduler=None):
        since = time.time()
        if scheduler is not None:
            scheduler = scheduler(self.optim, num_epochs)
        else:
            scheduler = Scheduler(self.optim, num_epochs)

        val_acc_history = []

        if state_dict is not None:
            self.load(self.models_dir / state_dict)
        best_acc = 0
        self.model.to(device)
        for epoch in range(num_epochs):
            epoch_start = time.time()
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_acc = 0
                k = 0

                with tqdm(dls[phase],
                          desc=f'epoch {epoch+1}/{num_epochs}: {phase}',
                          postfix={1: 0, 'loss': 0, 'acc': 0},
                          bar_format=('{n}/|/{l_bar}| {n_fmt}/{total_fmt}'
                                      ' [{elapsed}<{remaining}, {rate_fmt}], '
                                      'loss: {postfix[loss]}, '
                                      'acc: {postfix[acc]}')) as t:
                    for input, target in dls[phase]:
                        input = input.to(device)
                        target = target.to(device)

                        self.optim.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            output = self.model(input)
                            loss = self.loss(output, target)

                            if phase == 'train':
                                loss.backward()
                                self.optim.step()
                                if scheduler.step_on_batch:
                                    scheduler.step()

                        running_loss += loss.item()
                        acc = np.mean([metric(target, output).item()\
                                           for metric in self.metrics])
                        running_acc += acc
                        k += 1
                        t.postfix["loss"] = running_loss/k
                        t.postfix["acc"] = running_acc/k
                        t.update()

                epoch_loss = running_loss / k
                epoch_acc = running_acc / k

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    self.save(self.models_dir / save_name)
                    time_elapsed = time.time()-epoch_start
                    print((f'{epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} '
                           f'Acc: {epoch_acc:.4f} '
                           f'Dur: {timedelta(seconds=time_elapsed)}'))
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                else:
                    if not scheduler.step_on_batch:
                        scheduler.step()
            print()

        time_elapsed = time.time() - since
        print(
            f'Training complete in {timedelta(seconds=time_elapsed)}')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.load(self.models_dir / save_name)
        return self.model, val_acc_history

    def score(self, dl, device):
        loss_tot = 0
        metrics_tot = {metric: 0 for metric in self.metrics}
        for input, target in dl:
            input = input.to(device)
            target = target.to(device)
            with torch.no_grad():
                output = self.model(input)
                loss_tot += self.loss(output, target).item()
                for metric in self.metrics:
                    metrics_tot[metric] += metric(target, output).item()
        loss_tot /= len(dl)
        s = [f'loss: {loss_tot:.4f}']
        for metric in metrics_tot:
            metrics_tot[metric] /= len(dl)
            s.append[f'{metric.__name__}: {metrics_tot[metric]:.4f}']
        print('; '.join(s))
        return loss_tot, metrics_tot

    def predict(self, dl, device, sizes, TTA=True, **kwargs):
        if TTA:
            return predict_TTA_all(self.model, dl, device, sizes, **kwargs)
        else:
            return predict_all(self.model, dl)

    def load(self, model):
        self.model.load_state_dict(torch.load(model))

    def save(self, path):
        torch.save(self.model.state_dict(), path)


class Scheduler:
    def __init__(
            self, optim=None, n_epochs=None, last_step=-1,
            step_on_batch=False):
        self.optim = optim
        self.n_epochs = None
        self.last_step = last_step
        self.step_on_batch = step_on_batch

    def __call__(self, optim, n_epochs):
        self.optim = optim
        self.n_epochs = n_epochs
        return self

    def get_lr(self):
        return [group['lr'] for group in self.optim.param_groups]

    def step(self, epoch=None):
        assert self.optim is not None, 'An optimizer must be specified'
        assert self.n_epochs is not None, 'A number of epochs must be specified'
        if epoch is None:
            epoch = self.last_step + 1
        self.last_step = epoch
        for param_group, lr in zip(self.optim.param_groups, self.get_lr()):
            param_group['lr'] = lr


class OneCycleScheduler(Scheduler):
    def __init__(
            self, lr_max, train_len, div_factor=25, moms=(0.95, 0.85),
            pct_start=0.3, final_div=1e4, bs=8, **kwargs):
        super().__init__(step_on_batch=True, **kwargs)
        if isinstance(lr_max, Number):
            lr_max = [lr_max]
        if self.optim is not None:
            lr_max *= len(self.optim.param_groups)
        self.lr_max = lr_max
        self.lr_min = [lr/div_factor for lr in lr_max]
        self.mom_max = moms[0]
        self.mom_min = moms[1]
        self.final_div = final_div
        n = math.ceil(train_len/bs)
        self.a1 = pct_start*n
        self.a2 = n-self.a1
    
    def __call__(self, optim, n_epochs):
        super().__call__(optim, n_epochs)
        self.lr_max *= len(optim.param_groups)
        self.lr_min *= len(optim.param_groups)
        self.a1 *= n_epochs
        self.a2 *= n_epochs
        return self
    
    def get_lr(self):
        if self.last_step <= self.a1:
            return [an.annealing_cos(
                lr1, lr2, self.last_step/self.a1) for lr1, lr2 in zip(
                self.lr_min, self.lr_max)]
        else:
            return [an.annealing_cos(
                lr1, lr2,
                (self.last_step-self.a1)/self.a2) for lr1, lr2 in zip(
                self.lr_max, self.lr_min)]

    def get_mom(self):
        if self.last_step <= self.a1:
            return an.annealing_cos(
                self.mom_max, self.mom_min, self.last_step/self.a1)
        else:
            return an.annealing_cos(
                self.mom_min, self.mom_max, (self.last_step-self.a1)/self.a2)

    def step(self, epoch=None):
        super().step(epoch)
        mom = self.get_mom()
        for param_group in self.optim.param_groups:
            param_group['momentum'] = mom
