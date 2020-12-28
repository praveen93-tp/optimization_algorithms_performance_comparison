import torch
import numpy as np
from torch.optim import Optimizer
from torch.distributions import Bernoulli, Normal
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from collections import defaultdict
import math


class Adam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, nesterov=False, l2_reg=0, weight_decay=0, rectified=False, lrd=1, eps=1e-8):
        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'nesterov': nesterov, 'l2_reg': l2_reg,
                    'weight_decay': weight_decay, 'rectified': rectified, 'lrd': lrd, 'eps': eps}
        super(Adam, self).__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            nesterov = group['nesterov']
            l2_reg = group['l2_reg']
            weight_decay = group['weight_decay']
            rectified = group['rectified']
            lrd_bernoulli = Bernoulli(probs=group['lrd'])
            eps = group['eps']
            if 'm' not in group and 'v' not in group:
                group['m'] = []
                group['v'] = []
                group['t'] = 1
                if nesterov:
                    group['prev_grad'] = []
                for param in group['params']:
                    group['m'].append(torch.zeros_like(param))
                    group['v'].append(torch.zeros_like(param))
                    if nesterov:
                        group['prev_grad'].append(torch.zeros_like(param))
            for idx, param in enumerate(group['params']):
                if l2_reg:
                    param.grad.data += l2_reg * param.data
                if nesterov:
                    grad = group['prev_grad'][idx]
                else:
                    grad = param.grad.data
                lrd_mask = lrd_bernoulli.sample(param.size()).to(param.device)
                m = group['m'][idx]
                v = group['v'][idx]
                t = group['t']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                if nesterov:
                    group['prev_grad'][idx] = param.grad.data
                if rectified:
                    rho_inf = 2 / (1 - beta2) - 1
                    rho = rho_inf - 2 * t * beta2**t / (1 - beta2**t)
                    if rho >= 5:
                        numerator = (1 - beta2**t) * (rho - 4) * (rho - 2) * rho_inf
                        denominator = (rho_inf - 4) * (rho_inf - 2) * rho
                        r = np.sqrt(numerator / denominator)
                        param.data += - lrd_mask * lr * r * m_hat / (torch.sqrt(v) + eps)
                    else:
                        param.data += - lrd_mask * lr * m_hat
                else:
                    param.data += - lrd_mask * lr * m_hat / (torch.sqrt(v_hat) + eps)
                if weight_decay:
                    param.data -= weight_decay * param.data
                group['m'][idx] = m
                group['v'][idx] = v
            group['t'] += 1



class Padam(Optimizer):
    def __init__(self, params, lr=1e-1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True, partial=1 / 4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, partial=partial)
        super(Padam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsgrad = group['amsgrad']
                partial = group['partial']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'],p.data)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom ** (partial * 2))
        return loss


class SGD(Optimizer):
    def __init__(self, params, lr, mu=0, nesterov=False, weight_decay=0, lrd=1):
        defaults = {'lr': lr, 'mu': mu, 'nesterov': nesterov, 'weight_decay': weight_decay, 'lrd': lrd}
        super(SGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            lrd_bernoulli = Bernoulli(probs=group['lrd'])
            if mu != 0 and 'v' not in group:
                group['v'] = []
                if nesterov:
                    group['theta'] = []
                for param in group['params']:
                    group['v'].append(torch.zeros_like(param))
                    if nesterov:
                        theta_param = torch.ones_like(param).mul_(param.data)
                        group['theta'].append(theta_param)
            for idx, param in enumerate(group['params']):
                param.grad.data -= weight_decay * param.data
                lrd_mask = lrd_bernoulli.sample(param.size()).to(param.device)
                if mu != 0:
                    v = group['v'][idx]
                    v = mu * v - lr * param.grad.data
                    group['v'][idx] = v
                    if nesterov:
                        group['theta'][idx] += lrd_mask * v
                        param.data = group['theta'][idx] + mu * v
                    else:
                        param.data += lrd_mask * v
                else:
                    param.data -= lrd_mask * lr * param.grad.data



class RMSProp(Adam):
    def __init__(self, params, lr, beta2):
        super(RMSProp, self).__init__(params, lr, beta2=beta2, beta1=0)


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups

        self.counter = 0
        for group in optimizer.param_groups:
            group['phi'] = []
            for param in group['params']:
                phi_param = torch.ones_like(param).mul_(param.data)
                group['phi'].append(phi_param)
    def step(self):
        if self.counter == self.k:
            for group_idx, group in enumerate(self.param_groups):
                for idx, _ in enumerate(group['phi']):
                    theta = self.optimizer.param_groups[group_idx]['params'][idx].data
                    group['phi'][idx] = group['phi'][idx] + self.alpha * (theta - group['phi'][idx])
            self.counter = 0
        else:
            self.counter += 1
            self.optimizer.step()


class GradNoise(Optimizer):
    def __init__(self, optimizer, eta=0.3, gamma=0.55):
        self.optimizer = optimizer
        self.eta = eta
        self.gamma = gamma
        self.t = 0
        self.param_groups = optimizer.param_groups
    def step(self):
        normal = torch.empty(1).normal_(mean=0, std=np.sqrt(self.eta/((1+self.t)**self.gamma)))\
            .to(self.optimizer.param_groups[0]['params'][0].device)
        for group_idx, group in enumerate(self.param_groups):
            for idx, param in enumerate(group['params']):
                self.optimizer.param_groups[group_idx]['params'][idx].grad.data += normal
                self.optimizer.step()
                self.t += 1



class GradDropout(Optimizer):
    def __init__(self, optimizer, grad_retain=0.9):
        self.optimizer = optimizer
        self.grad_retain = grad_retain
        self.grad_bernoulli = Bernoulli(probs=grad_retain)
        self.param_groups = optimizer.param_groups
    def step(self):
        for group_idx, group in enumerate(self.param_groups):
            for idx, param in enumerate(group['params']):
                grad_mask = self.grad_bernoulli.sample(param.size()).to(param.device)
                self.optimizer.param_groups[group_idx]['params'][idx].grad.data *= grad_mask
                self.optimizer.step()


class AMSGrad(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AMSGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AMSGrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                amsgrad = group['amsgrad']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

class Adamax(Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adamax, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_inf'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                beta1, beta2 = group['betas']
                eps = group['eps']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                norm_buf = torch.cat([
                    exp_inf.mul_(beta2).unsqueeze(0),
                    grad.abs().add_(eps).unsqueeze_(0)
                ], 0)
                torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                bias_correction = 1 - beta1 ** state['step']
                clr = group['lr'] / bias_correction
                p.addcdiv_(exp_avg, exp_inf, value=-clr)
        return loss

class AdaBound(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,eps=1e-8, weight_decay=0, amsbound=False):
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))
    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsbound = group['amsbound']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'],1,p.data)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)
                p.data.add_(-step_size)

        return loss

class AdaBoundW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,eps=1e-8, weight_decay=0, amsbound=False):
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBoundW, self).__init__(params, defaults)
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))
    def __setstate__(self, state):
        super(AdaBoundW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsbound = group['amsbound']

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)
                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.add_(-step_size)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.add_(-step_size)
        return loss