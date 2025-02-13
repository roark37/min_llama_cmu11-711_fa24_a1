from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]

                # init state at the beginning
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Access hyperparameters from the `group` dictionary
                beta1, beta2 = group['betas']
                alpha = group["lr"] # alpha is the stepsize
                eps = group['eps']
                weight_decay = group['weight_decay']
                correct_bias = group['correct_bias']


                # access the old states
                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Update first and second moments of the gradients
                # --> use inplace ops for efficiency
                # same as: 
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad 
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # same as: 
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # exp_avg_sq_eps = exp_avg_sq.sqrt() + eps
                exp_avg_sq_eps = exp_avg_sq.sqrt().add_(eps)

                # Bias correction
                if correct_bias:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    alpha *= (bias_correction2 ** 0.5 / bias_correction1)

                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980

                # Update parameters
                # --> use inplace ops for efficiency
                # same as: 
                # p.data -= alpha * exp_avg / exp_avg_sq_eps
                p.data.addcdiv_(exp_avg, exp_avg_sq_eps, value=-alpha)

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                # --> This is where Adam is different from AdamW:
                # Using lr in both places ensures that when change the lr, 
                # both the gradient-based updates and weight decay updates scale proportionally. 
                # If not, then:
                # 1. At high lr: weight decay effect becomes relatively insignificant
                # 2. At low lr: weight decay would dominate the updates
                if weight_decay != 0:
                    # same as: 
                    # p.data -= alpha * weight_decay
                    p.data.add_(p.data, alpha=-alpha * weight_decay)

        return loss