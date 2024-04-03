import logging
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List

import torch
import torch.distributed as dist
import torch.optim

class AdamEF(torch.optim.Optimizer):
    r"""Adapt Adam for Embedding Resettting
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        lr_emb=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        clear_embed_every_K_updates=0,
        embed_offset=None,
        stop_after=None,
    ):
        defaults = dict(lr=lr, lr_emb=lr_emb, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamEF, self).__init__(params, defaults)
        self.clear_embed_every_K_updates = clear_embed_every_K_updates
        self.embed_offset = embed_offset
        self.stop_after = stop_after

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def get_step_emb(self):
        """return state['step_emb'], assume step_emb is the same for each param"""
        p = self.param_groups[0]['params'][0]
        if ['step_emb'] in self.state[p]:
            return self.state[p]['step_emb']
        else:
            return 0 # init value of step_emb

    def set_lr_emb(self, lr_emb):
        for param_group in self.param_groups:
            param_group["lr_emb"] = lr_emb

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group.get("amsgrad", False)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["step_emb"] = 0 # counter for effective embedding updates
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    assert self.clear_embed_every_K_updates > 0 #TODO; refactor into some function
                    if self.stop_after is not None and state["step"] < self.stop_after:
                        if state["step"] % self.clear_embed_every_K_updates == 0:
                            state["step_emb"] = 0 # reset counter for effective embedding updates
                            state["exp_avg"][0: self.embed_offset].fill_(0) # reset the running sum (emb) to 0
                            state["exp_avg_sq"][0: self.embed_offset].fill_(0)

                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                            p_data_fp32
                        )
                # logger.info("Step {}, Step emb {}, LR {}, LR emb {}".format(state['step'], state['step_emb'], group['lr'], group['lr_emb']))
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                state["step_emb"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                bias_correction1_emb = 1 - beta1 ** state["step_emb"]
                bias_correction2_emb = 1 - beta2 ** state["step_emb"]
                
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                step_size_emb = group["lr_emb"] * math.sqrt(bias_correction2_emb) / bias_correction1_emb 

                if group["weight_decay"] != 0:
                    p_data_fp32[self.embed_offset:].add_(
                        p_data_fp32[self.embed_offset:], alpha=-group["weight_decay"] * group["lr"]
                    )
                    p_data_fp32[0:self.embed_offset].add_(
                        p_data_fp32[0:self.embed_offset], alpha=-group["weight_decay"] * group["lr_emb"]
                    )

                p_data_fp32[self.embed_offset:].addcdiv_(exp_avg[self.embed_offset:], denom[self.embed_offset:], value=-step_size)
                p_data_fp32[0:self.embed_offset].addcdiv_(exp_avg[0:self.embed_offset], denom[0:self.embed_offset], value=-step_size_emb)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)
        return loss