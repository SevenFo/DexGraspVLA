from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import inspect
from dexgraspvla.controller.model.common.normalizer import LinearNormalizer
from dexgraspvla.controller.policy.base_image_policy import BaseImagePolicy
from dexgraspvla.controller.model.diffusion.transformer_for_action_diffusion import (
    TransformerForActionDiffusion,
)
from dexgraspvla.controller.model.vision.obs_encoder import ObsEncoder
from scipy.optimize import linear_sum_assignment
import pickle


# Adapted from https://github.com/lucidrains/pi-zero-pytorch/blob/e82fced40e55023a0ded22ab3bda495964353253/pi_zero_pytorch/pi_zero.py#L216
def noise_assignment(data, noise):
    device = data.device
    data, noise = tuple(rearrange(t, "b ... -> b (...)") for t in (data, noise))
    dist = torch.cdist(data, noise)
    _, assign = linear_sum_assignment(dist.cpu())
    return torch.from_numpy(assign).to(device)


class DexGraspVLAController(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        # 删除 noise_scheduler: DDPMScheduler 参数
        obs_encoder: ObsEncoder,
        num_inference_steps=100,  # 默认值，控制 ODE 求解步数
        # arch
        n_layer=7,
        n_head=8,
        p_drop_attn=0.1,
        use_attn_mask=False,
        start_ckpt_path=None,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # 解析形状
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta["action"]["horizon"]

        obs_shape, obs_part_length = obs_encoder.output_shape()
        n_emb = obs_shape[-1]
        obs_tokens = obs_shape[-2]

        model = TransformerForActionDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens + 1,  # obs tokens + 1 token for time
            p_drop_attn=p_drop_attn,
            obs_part_length=obs_part_length,
            use_attn_mask=use_attn_mask,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        # 移除 self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.start_ckpt_path = start_ckpt_path
        self.kwargs = kwargs

        # 设置 ODE 求解步数
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self, cond=None, gen_attn_map=True, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler
        B = cond.shape[0]

        trajectory = torch.randn(
            size=(B, self.action_horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        # Store attention maps for all timesteps
        all_timestep_attention_maps = {}

        for t in scheduler.timesteps:
            # 1. predict model output
            model_output, attention_maps = model(
                trajectory, t, cond, training=False, gen_attn_map=gen_attn_map
            )
            all_timestep_attention_maps[t.cpu().item()] = attention_maps

            # 2. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, **kwargs
            ).prev_sample

        return trajectory, all_timestep_attention_maps

    def solve_ode(self, cond, gen_attn_map=False, **kwargs):
        """
        使用欧拉方法求解 ODE dx/dt = v(x,t)
        从 t=1 (噪声) 积分到 t=0 (数据)
        """
        model = self.model
        B = cond.shape[0]

        # 1. 从 t=1 处的纯高斯噪声开始
        x_t = torch.randn(
            size=(B, self.action_horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
        )

        # 2. 定义 ODE 求解器的时间步长
        ts = torch.linspace(1, 0, self.num_inference_steps, device=self.device)

        # 存储注意力图（如果需要的话）
        all_timestep_attention_maps = {}

        # 3. ODE 积分循环
        for i in range(self.num_inference_steps - 1):
            t_now = ts[i]
            t_next = ts[i + 1]

            # 将 t_now 广播为模型需要的形状
            t_now_tensor = t_now.expand(B)

            # 预测向量场 v(x_t, t)
            with torch.no_grad():
                v_pred, attention_maps = model(
                    sample=x_t,
                    timestep=t_now_tensor,
                    cond=cond,
                    training=False,
                    gen_attn_map=gen_attn_map,
                )

            if gen_attn_map:
                all_timestep_attention_maps[t_now.cpu().item()] = attention_maps

            # 4. 应用欧拉方法的一步积分
            # x_next = x_now + v * dt，其中 dt = t_next - t_now（是负值）
            dt = t_next - t_now
            x_t = x_t + v_pred * dt

        # t=0 时的 x_t 就是我们生成的样本
        return x_t, all_timestep_attention_maps

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], output_path: str = None
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: 必须包含 "obs" 键
        action_pred: 预测的动作
        """
        assert "past_action" not in obs_dict  # 尚未实现
        nobs = obs_dict
        B = next(iter(nobs.values())).shape[0]

        # 处理输入
        obs_tokens = self.obs_encoder(nobs, training=False)

        # 通过求解 ODE 运行采样
        nsample, all_timestep_attention_maps = self.solve_ode(
            cond=obs_tokens,
            gen_attn_map=True if output_path is not None else False,
            **self.kwargs,
        )

        # 反归一化预测
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer["action"].unnormalize(nsample)

        if output_path is not None:
            # Convert tensors in obs_dict to numpy arrays
            obs_dict_numpy = {}
            for k, v in obs_dict.items():
                if k in ["rgbm", "right_cam_img"]:
                    obs_dict_numpy[k] = np.clip(
                        v.detach().cpu().numpy() * 255, 0, 255
                    ).astype(np.uint8)
                else:
                    obs_dict_numpy[k] = v.detach().cpu().numpy()
                obs_dict_numpy[k] = obs_dict_numpy[k][:2]

            save_dict = {
                "attention_maps": all_timestep_attention_maps,
                "obs_dict": obs_dict_numpy,
            }

            with open(output_path, "wb") as f:
                pickle.dump(save_dict, f)

        return action_pred

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        print(f"Fused AdamW available: {fused_available}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas, fused=fused_available
        )
        return optimizer

    def compute_loss(self, batch, training=True):
        # normalize input
        assert "valid_mask" not in batch
        nobs = batch["obs"]

        # 1. 获取干净数据 (x1) 和先验噪声 (x0)
        x1 = self.normalizer["action"].normalize(batch["action"])
        x0 = torch.randn_like(x1)

        # 处理观察向量
        obs_tokens = self.obs_encoder(nobs, training)

        # 2. 从 [0, 1] 均匀采样时间 t (避免精确为0的情况)
        t = torch.rand(x1.shape[0], device=x1.device) * (1 - 1e-4) + 1e-4

        # 调整 t 的形状以便广播: (B, 1, 1)
        time_expanded = t.view(-1, 1, 1)

        # 3. 创建插值样本 x_t = (1-t) * x1 + t * x0
        x_t = (1 - time_expanded) * x1 + time_expanded * x0

        # 4. 定义目标向量场 u_t = x0 - x1
        u_t = x0 - x1

        # 5. 使用模型预测向量场 v_t
        v_t_pred, _ = self.model(
            sample=x_t,
            timestep=t,  # 直接传递 t，现在是 [0,1] 范围内的浮点数
            cond=obs_tokens,
            training=training,
            gen_attn_map=False,
        )

        # 6. 计算预测向量场与目标向量场之间的 MSE 损失
        loss = F.mse_loss(v_t_pred, u_t)

        return loss

    def forward(self, batch, training=True):
        return self.compute_loss(batch, training)
