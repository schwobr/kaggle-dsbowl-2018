from tqdm.autonotebook import tqdm
from datetime import timedelta
import os
import time
import torch


class Net:
    def __init__(self, model, optim, loss, metrics, models_dir):
        self.model = model
        self.optim = optim
        self.loss = loss
        self.metrics = metrics
        self.models_dir = models_dir

    def __call__(self, input):
        return self.model(input)

    def fit(self, dls, num_epochs, save_name, device, state_dict=None):
        since = time.time()

        val_acc_history = []

        if state_dict:
            self.model.load_state_dict(
                os.path.join(self.models_dir, state_dict))
        best_acc = 0

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

                        running_loss += loss.item()
                        acc = torch.mean([(torch.mean(
                                          metric(target.cpu(), output.cpu()))
                            for metric in self.metrics)])
                        running_acc += acc.item()
                        k += 1
                        t.postfix["loss"] = running_loss/k
                        t.postfix["acc"] = running_acc/k
                        t.update()

                epoch_loss = running_loss / k
                epoch_acc = running_acc / k

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    self.model.save(os.path.join(self.models_dir, save_name))
                    time_elapsed = time.time()-epoch_start
                    print((f'{epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} '
                           f'Acc: {epoch_acc:.4f} '
                           f'Dur: {timedelta(seconds=time_elapsed)}'))
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
            print()

        time_elapsed = time.time() - since
        print(
            f'Training complete in {timedelta(seconds=time_elapsed)}')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(torch.load(
            os.path.join(self.models_dir, save_name)))
        return self.model, val_acc_history
