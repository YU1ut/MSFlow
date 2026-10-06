from __future__ import annotations

import argparse
import random
import sys
from os.path import join as pjoin
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

# Support direct execution from the repository root:
#   uv run python sample/demo_joint_contorl.py ...
_this = Path(__file__).absolute()


def _find_project_root() -> Path:
    starts = [Path.cwd().absolute(), _this.parent]
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "models" / "MMDiT_xyz.py").is_file() and (
                candidate / "train" / "train_msflow_xyz.py"
            ).is_file():
                return candidate
    raise RuntimeError(
        "Could not find the project root containing the raw MMDiT model."
    )


_project_root = _find_project_root()
_project_root_str = str(_project_root)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# isort: off
from utils.raw_inpaint_common import (  # noqa: E402
    dit_norm_to_world,
    load_sparse_raw_control_spec,
    raw_control_chw_to_dit,
    resolve_value,
    save_raw_inpaint_outputs,
    str2bool,
)
from utils.raw_inpaint_model import load_model  # noqa: E402
from jit_diffusions.transport import path as transport_path  # noqa: E402

from ProjFlow.diffusions.transport.projflow_helpers import (  # noqa: E402
    KinematicMetric,
    build_kinematic_metric,
    build_pseudo_observations,
    build_skeleton_laplacian,
    curvature_per_frame,
    distribute_trust_over_halo_joints,
    flowdps_eta,
    frame_trust_schedule,
    halo_radius_linear,
    metric_project_clean_endpoint,
    trust_to_variance,
    tweedie_endpoints_from_velocity,
)
from utils.config_utils import load_yaml_config  # noqa: E402
from utils.raw_joint_utils import load_raw_joint_mean_std  # noqa: E402
from utils.train_utils import lengths_to_mask  # noqa: E402

# isort: on


def _dit_to_joint_layout(x: torch.Tensor, joint_count: int) -> torch.Tensor:
    """Convert [B, T, 1, J*3] Direct-DiT tensors to [B, 3, T, J]."""
    batch_size, n_frames, singleton, input_dim = x.shape
    if singleton != 1 or input_dim != joint_count * 3:
        raise ValueError(
            f"Expected [B, T, 1, {joint_count * 3}], got {tuple(x.shape)}."
        )
    return (
        x[:, :, 0]
        .reshape(batch_size, n_frames, joint_count, 3)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def _joint_to_dit_layout(x: torch.Tensor) -> torch.Tensor:
    """Convert [B, 3, T, J] raw-joint tensors to [B, T, 1, J*3]."""
    batch_size, channels, n_frames, joint_count = x.shape
    if channels != 3:
        raise ValueError(f"Expected three XYZ channels, got {channels}.")
    return (
        x.permute(0, 2, 3, 1)
        .contiguous()
        .reshape(batch_size, n_frames, 1, joint_count * channels)
    )


def _uses_joint_layout(x: torch.Tensor, joint_count: int) -> bool:
    return x.dim() == 4 and x.shape[1] == 3 and x.shape[3] == joint_count


def _to_joint_layout(x: torch.Tensor, joint_count: int) -> torch.Tensor:
    if _uses_joint_layout(x, joint_count):
        return x
    return _dit_to_joint_layout(x, joint_count)


def _identity_metric(laplacian: torch.Tensor) -> KinematicMetric:
    ones = torch.ones(
        laplacian.shape[0],
        device=laplacian.device,
        dtype=laplacian.dtype,
    )
    return KinematicMetric(
        apply_Rinv=lambda value: value,
        diag_Rinv_joint=ones,
        joint_weights_q=ones,
        L_kin=laplacian,
    )


def _build_augmented_observations(
    hard_mask: torch.Tensor,
    hard_value: torch.Tensor,
    halo_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build pseudo-observations independently for each controlled axis."""
    pseudo_values = []
    halo_masks = []
    for axis in range(hard_mask.shape[1]):
        pseudo_value, halo_mask = build_pseudo_observations(
            hard_mask=hard_mask[:, axis : axis + 1],
            hard_value=hard_value[:, axis : axis + 1],
            halo_radius=halo_radius,
        )
        pseudo_values.append(pseudo_value)
        halo_masks.append(halo_mask)
    return torch.cat(pseudo_values, dim=1), torch.cat(halo_masks, dim=1)


def _project_per_axis(
    *,
    x1_hat: torch.Tensor,
    selector: torch.Tensor,
    targets: torch.Tensor,
    metric: KinematicMetric,
    sigma2: torch.Tensor | None,
    schur_block: int,
) -> torch.Tensor:
    """Project XYZ independently so --axes masks remain axis-specific."""
    projected = []
    for axis in range(x1_hat.shape[1]):
        axis_sigma2 = None if sigma2 is None else sigma2[:, axis : axis + 1]
        projected.append(
            metric_project_clean_endpoint(
                x1_hat=x1_hat[:, axis : axis + 1],
                selector=selector[:, axis : axis + 1],
                targets=targets[:, axis : axis + 1],
                apply_Rinv=metric.apply_Rinv,
                sigma2=axis_sigma2,
                block_size=schur_block,
            )
        )
    return torch.cat(projected, dim=1)


def _noise_mix(
    x0_hat: torch.Tensor,
    eta: torch.Tensor,
    noise_scale: float,
    strength: float | torch.Tensor,
) -> torch.Tensor:
    if noise_scale == 0.0:
        return x0_hat

    # Preserve the model's training prior. Strength 1 matches the absolute
    # perturbation of the unit-variance FlowDPS schedule; larger values refresh
    # the high-variance prior more aggressively.
    refresh_variance = max(noise_scale * noise_scale, 1.0)
    refresh_eta = torch.clamp(
        eta * strength / refresh_variance,
        min=0.0,
        max=1.0,
    )
    eps = torch.randn_like(x0_hat) * noise_scale
    return torch.sqrt(1.0 - refresh_eta) * x0_hat + torch.sqrt(refresh_eta) * eps


def _prediction_endpoints(
    *,
    x_t: torch.Tensor,
    t: torch.Tensor,
    model_output: torch.Tensor,
    prediction_output: str,
    path_sampler: Any,
    sigma_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction_output == "velocity":
        return tweedie_endpoints_from_velocity(
            x_t=x_t,
            t=t,
            v=model_output,
            path_sampler=path_sampler,
        )
    if prediction_output != "x":
        raise ValueError(f"Unsupported flow prediction output: {prediction_output}.")

    # The model directly predicts the clean endpoint. Converting that prediction
    # to velocity with a sigma_min-clamped denominator and then inverting it
    # biases x1_hat whenever sigma_t < sigma_min (for example, the final update
    # of a 200-step grid). Keep the clean prediction exact and recover only x0.
    t_like = transport_path.expand_t_like_x(t, x_t)
    alpha_t, _ = path_sampler.compute_alpha_t(t_like)
    sigma_t, _ = path_sampler.compute_sigma_t(t_like)
    x1_hat = model_output
    x0_hat = (x_t - alpha_t * x1_hat) / sigma_t.clamp_min(sigma_min)
    return x1_hat, x0_hat


def _projected_raw_step(
    *,
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    model_output: torch.Tensor,
    prediction_output: str,
    hard_value: torch.Tensor,
    hard_mask: torch.Tensor,
    path_sampler: Any,
    metric: KinematicMetric,
    joint_count: int,
    step: int,
    num_steps: int,
    use_metric_R: bool,
    use_augmented_obs: bool,
    use_noise_mixing: bool,
    noise_scale: float,
    noise_mix_strength: float | torch.Tensor,
    sigma_min: float,
    w_kin: float,
    ridge: float,
    ell_min: float,
    ell_max: float,
    tau_min: float,
    c0: float,
    lambda_s: float,
    trust_power: float,
    pi_min: float,
    pi_max: float,
    schur_block: int,
) -> torch.Tensor:
    x1_hat_model, x0_hat = _prediction_endpoints(
        x_t=x_t,
        t=t,
        model_output=model_output,
        prediction_output=prediction_output,
        path_sampler=path_sampler,
        sigma_min=sigma_min,
    )
    uses_joint_layout = _uses_joint_layout(x_t, joint_count)
    x1_hat = _to_joint_layout(x1_hat_model, joint_count)

    if use_augmented_obs:
        halo_radius = halo_radius_linear(
            step,
            num_steps,
            ell_min=ell_min,
            ell_max=ell_max,
        )
        pseudo_value, halo_mask = _build_augmented_observations(
            hard_mask,
            hard_value,
            halo_radius,
        )
        selector = ((hard_mask > 0.5) | (halo_mask > 0.5)).to(x1_hat.dtype)
        targets = torch.where(
            hard_mask > 0.5,
            hard_value,
            torch.where(halo_mask > 0.5, pseudo_value, hard_value),
        )
        curvature = curvature_per_frame(
            x1_hat=x1_hat,
            metric=metric,
            w_kin=w_kin,
            ridge=ridge,
        )
        frame_trust = frame_trust_schedule(
            t_scalar=float(t[0].item()),
            curvature=curvature,
            pi_min=pi_min,
            pi_max=pi_max,
            tau_min=tau_min,
            c0=c0,
            lambda_s=lambda_s,
            p=trust_power,
        )
        joint_trust = distribute_trust_over_halo_joints(
            pi_frame=frame_trust,
            M_halo_any=halo_mask.amax(dim=1) > 0.5,
            q_joint=metric.joint_weights_q,
            pi_min=pi_min,
            pi_max=pi_max,
        )
        sigma2 = trust_to_variance(
            pi_joint=joint_trust,
            metric=metric,
            hard_mask=hard_mask,
            M_all=selector,
        )
    else:
        selector = (hard_mask > 0.5).to(x1_hat.dtype)
        targets = hard_value
        sigma2 = None

    if not use_metric_R and not use_augmented_obs:
        x1_proj = torch.where(selector > 0.5, targets, x1_hat)
    else:
        x1_proj = _project_per_axis(
            x1_hat=x1_hat,
            selector=selector,
            targets=targets,
            metric=metric,
            sigma2=sigma2,
            schur_block=schur_block,
        )
    x1_proj_model = x1_proj if uses_joint_layout else _joint_to_dit_layout(x1_proj)

    alpha_next, _ = path_sampler.compute_alpha_t(
        transport_path.expand_t_like_x(t_next, x_t)
    )
    sigma_next, _ = path_sampler.compute_sigma_t(
        transport_path.expand_t_like_x(t_next, x_t)
    )
    if use_noise_mixing:
        x0_hat = _noise_mix(
            x0_hat,
            flowdps_eta(sigma_next),
            noise_scale,
            noise_mix_strength,
        )
    return alpha_next * x1_proj_model + sigma_next * x0_hat


@torch.no_grad()
def sample_batch_with_raw_dit_inpaint(
    model: torch.nn.Module,
    *,
    prompts: Sequence[str],
    control: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    cfg: float,
    num_steps: int,
    joint_count: int = 22,
    use_metric_R: bool = True,
    use_augmented_obs: bool = True,
    use_noise_mixing: bool = True,
    noise_mix_strength: float | None = None,
    w_kin: float = 10.0,
    ridge: float = 1.0,
    ell_min: float = 3.0,
    ell_max: float = 10.0,
    tau_min: float = 0.1,
    c0: float = 3.0,
    lambda_s: float = 1.0,
    trust_power: float = 2.0,
    pi_min: float = 0.02,
    pi_max: float = 1.0,
    schur_block: int = 1024,
) -> torch.Tensor:
    """Sample a batch of raw 22x3 Direct-DiT motions with ProjFlow."""
    if num_steps < 2:
        raise ValueError("--num_steps must be at least 2.")
    if noise_mix_strength is not None and noise_mix_strength < 0.0:
        raise ValueError("noise_mix_strength must be non-negative.")

    device = next(model.parameters()).device
    batch_size, n_frames, _, input_dim = control.shape
    if len(prompts) != batch_size or mask.shape != control.shape:
        raise ValueError("Prompts, control, and mask must have the same batch size.")
    if input_dim != joint_count * 3:
        raise ValueError(
            f"Raw model input_dim must be {joint_count * 3}, got {input_dim}."
        )

    control = control.to(device=device, dtype=torch.float32)
    mask = mask.to(device=device, dtype=torch.float32)
    lengths = lengths.to(device=device, dtype=torch.long)
    if lengths.shape != (batch_size,) or int(lengths.max().item()) > n_frames:
        raise ValueError(f"Expected {batch_size} lengths no greater than {n_frames}.")
    if hasattr(model, "scale_motion"):
        control = model.scale_motion(control)

    cond_vector, conds_mask = model.encode_text(list(prompts))
    cond_vector = cond_vector.to(device=device, dtype=torch.float32)
    conds_mask = conds_mask.to(device=device)
    attention_mask = lengths_to_mask(lengths, n_frames)
    padding_mask = ~attention_mask
    noise_scale = (
        model.resolve_noise_scale()
        if hasattr(model, "resolve_noise_scale")
        else float(getattr(model, "noise_scale", 1.0))
    )
    raw_joint_layout = bool(getattr(model, "raw_joint_layout", False))
    noise_shape = (
        (batch_size, 3, n_frames, joint_count)
        if raw_joint_layout
        else (batch_size, n_frames, 1, input_dim)
    )
    x = (
        torch.randn(
            *noise_shape,
            device=device,
        )
        * noise_scale
    )

    cfg_value = float(cfg)
    if cfg_value != 1.0:
        cond_vector = torch.cat(
            [cond_vector, torch.zeros_like(cond_vector)],
            dim=1,
        )
        conds_mask = torch.cat([conds_mask, conds_mask], dim=0)
        attention_mask = attention_mask.repeat(2, 1)
        x = torch.cat([x, x], dim=0)
        control = control.repeat(2, 1, 1, 1)
        mask = mask.repeat(2, 1, 1, 1)
    model_kwargs = {
        "conds": cond_vector,
        "conds_mask": conds_mask,
        "attention_mask": attention_mask,
        "cfg": cfg_value,
    }
    hard_value = _dit_to_joint_layout(control, joint_count)
    hard_mask = _dit_to_joint_layout(mask, joint_count)
    if noise_mix_strength is None:
        # One-frame controls already stay on-distribution. Repeated clean-endpoint
        # projections need more decorrelation for this model's 5x noise prior.
        controlled_frames = (hard_mask > 0.5).any(dim=1).any(dim=-1).sum(dim=-1)
        noise_mix_strength = torch.where(
            controlled_frames > 1,
            torch.full_like(controlled_frames, 4.0, dtype=x.dtype),
            torch.ones_like(controlled_frames, dtype=x.dtype),
        ).view(-1, 1, 1, 1)
    model_control = hard_value if raw_joint_layout else control
    model_mask = hard_mask if raw_joint_layout else mask

    laplacian = build_skeleton_laplacian(
        joint_count,
        device=device,
        dtype=x.dtype,
    )
    metric = (
        build_kinematic_metric(
            J=joint_count,
            w_kin=w_kin,
            ridge=ridge,
            L_kin=laplacian,
        )
        if use_metric_R
        else _identity_metric(laplacian)
    )

    sampler = model.gen_diffusion
    transport = sampler.transport
    path_sampler = transport.path_sampler
    t0, t1 = sampler.transport.check_interval(
        sampler.transport.train_eps,
        sampler.transport.sample_eps,
        sde=False,
        eval=True,
        reverse=False,
        last_step_size=0.0,
    )
    time_grid = torch.linspace(
        float(t0),
        float(t1),
        num_steps,
        device=device,
        dtype=x.dtype,
    )

    for step in range(num_steps - 1):
        t = torch.full(
            (x.shape[0],),
            float(time_grid[step].item()),
            device=device,
            dtype=x.dtype,
        )
        t_next = torch.full(
            (x.shape[0],),
            float(time_grid[step + 1].item()),
            device=device,
            dtype=x.dtype,
        )
        model_output = model.forward_with_CFG(
            x,
            t,
            **model_kwargs,
        )
        x = _projected_raw_step(
            x_t=x,
            t=t,
            t_next=t_next,
            model_output=model_output,
            prediction_output=sampler.transport.prediction_output,
            hard_value=hard_value,
            hard_mask=hard_mask,
            path_sampler=path_sampler,
            metric=metric,
            joint_count=joint_count,
            step=step,
            num_steps=num_steps,
            use_metric_R=use_metric_R,
            use_augmented_obs=use_augmented_obs,
            use_noise_mixing=use_noise_mixing,
            noise_scale=noise_scale,
            noise_mix_strength=noise_mix_strength,
            sigma_min=transport.sigma_min,
            w_kin=w_kin,
            ridge=ridge,
            ell_min=ell_min,
            ell_max=ell_max,
            tau_min=tau_min,
            c0=c0,
            lambda_s=lambda_s,
            trust_power=trust_power,
            pi_min=pi_min,
            pi_max=pi_max,
            schur_block=schur_block,
        )

    x = torch.where(model_mask > 0.5, model_control, x)
    if cfg_value != 1.0:
        x, _ = x.chunk(2, dim=0)
    if hasattr(model, "unscale_motion"):
        x = model.unscale_motion(x)
    if raw_joint_layout:
        x = _joint_to_dit_layout(x)
    return torch.where(
        padding_mask.unsqueeze(-1).unsqueeze(-1),
        torch.zeros_like(x),
        x,
    )


@torch.no_grad()
def sample_with_raw_dit_inpaint(
    model: torch.nn.Module,
    *,
    prompt: str,
    control: torch.Tensor,
    mask: torch.Tensor,
    length: int,
    num_samples: int,
    cfg: float,
    num_steps: int,
    seed: int,
    joint_count: int = 22,
    **projflow_kwargs: Any,
) -> torch.Tensor:
    """Sample repeated raw motions for the single-prompt demo."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    control = control.repeat(int(num_samples), 1, 1, 1)
    mask = mask.repeat(int(num_samples), 1, 1, 1)
    lengths = torch.full((int(num_samples),), int(length), dtype=torch.long)
    return sample_batch_with_raw_dit_inpaint(
        model,
        prompts=[prompt] * int(num_samples),
        control=control,
        mask=mask,
        lengths=lengths,
        cfg=cfg,
        num_steps=num_steps,
        joint_count=joint_count,
        **projflow_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ProjFlow sparse joint control for raw 22x3 MMDiT checkpoints "
            "trained by train/train_msflow_xyz.py"
        )
    )
    parser.add_argument("--dataset_name", "--dataset", default="t2m")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--checkpoints_dir", default="./checkpoints")
    parser.add_argument("--name", default="DiT_Direct_MM_Raw")
    parser.add_argument("--ckpt", default="latest.tar")
    parser.add_argument("--checkpoint_key", default="ema_model")
    parser.add_argument("--gpu", default="0")

    parser.add_argument("--n_frames", type=int, default=192)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--sigma_min", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--use_metric_R", type=str2bool, default=True)
    parser.add_argument("--use_augmented_obs", type=str2bool, default=True)
    parser.add_argument("--use_noise_mixing", type=str2bool, default=True)
    parser.add_argument("--noise_mix_strength", type=float, default=None)
    parser.add_argument(
        "--use_projflow",
        type=str2bool,
        default=None,
        help="Compatibility umbrella that sets all three ProjFlow toggles.",
    )
    parser.add_argument("--w_kin", type=float, default=10.0)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--ell_min", type=float, default=3.0)
    parser.add_argument("--ell_max", type=float, default=10.0)
    parser.add_argument("--tau_min", type=float, default=0.1)
    parser.add_argument("--c0", type=float, default=3.0)
    parser.add_argument("--lambda_s", type=float, default=1.0)
    parser.add_argument("--trust_power", type=float, default=2.0)
    parser.add_argument("--pi_min", type=float, default=0.02)
    parser.add_argument("--pi_max", type=float, default=1.0)
    parser.add_argument("--schur_block", type=int, default=1024)

    parser.add_argument(
        "--sparse_control_spec",
        required=True,
        help=("JSON specification containing arbitrary frame/joint target values."),
    )

    parser.add_argument("--out_dir", default="outputs/dit_mm_raw_inpaint")
    parser.add_argument("--save_mp4", action="store_true")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--raw_joint_count", type=int, default=22)
    parser.add_argument("--raw_joint_dim", type=int, default=3)
    parser.add_argument("--raw_mean_path", default=None)
    parser.add_argument("--raw_std_path", default=None)
    args = parser.parse_args()

    if args.dataset_name.lower() != "t2m":
        raise NotImplementedError("This raw Direct-DiT demo currently supports t2m.")
    if args.raw_joint_count != 22 or args.raw_joint_dim != 3:
        raise ValueError("Raw Direct-DiT inpainting expects 22 XYZ joints.")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.use_projflow is not None:
        args.use_metric_R = args.use_projflow
        args.use_augmented_obs = args.use_projflow
        args.use_noise_mixing = args.use_projflow

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.gpu != "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        if args.gpu != "cpu":
            print("CUDA is unavailable, falling back to CPU.")
        device = torch.device("cpu")

    run_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    saved_cfg = load_yaml_config(pjoin(run_dir, "config.yaml"))
    dataset_dir = str(resolve_value(args, saved_cfg, "dataset_dir", "./datasets"))
    data_root = pjoin(dataset_dir, "HumanML3D")
    mean, std = load_raw_joint_mean_std(
        args.dataset_name,
        data_root=data_root,
        mean_path=resolve_value(args, saved_cfg, "raw_mean_path", None),
        std_path=resolve_value(args, saved_cfg, "raw_std_path", None),
    )

    test_motion_id = None
    test_prompt_index = None
    sparse_control_metadata = None
    (
        control_chw,
        mask_chw,
        control_world,
        joint_ids,
        args.text,
        sparse_control_metadata,
    ) = load_sparse_raw_control_spec(
        args.sparse_control_spec,
        mean,
        std,
        n_joints=args.raw_joint_count,
    )
    args.n_frames = len(control_world)
    trajectory_used = control_world[:, list(joint_ids)]
    trajectory_requested = trajectory_used.copy()
    print(
        f"Using sparse frame/joint controls from {args.sparse_control_spec} "
        f"with {args.n_frames} frames."
    )

    model, input_dim = load_model(args, device=device)
    if input_dim != args.raw_joint_count * args.raw_joint_dim:
        raise ValueError(f"Raw model input_dim must be 66, got {input_dim}.")

    control, mask = raw_control_chw_to_dit(control_chw, mask_chw)
    samples_norm = sample_with_raw_dit_inpaint(
        model,
        prompt=args.text,
        control=control,
        mask=mask,
        length=args.n_frames,
        num_samples=args.num_samples,
        cfg=args.cfg,
        num_steps=args.num_steps,
        seed=args.seed,
        joint_count=args.raw_joint_count,
        use_metric_R=args.use_metric_R,
        use_augmented_obs=args.use_augmented_obs,
        use_noise_mixing=args.use_noise_mixing,
        noise_mix_strength=args.noise_mix_strength,
        w_kin=args.w_kin,
        ridge=args.ridge,
        ell_min=args.ell_min,
        ell_max=args.ell_max,
        tau_min=args.tau_min,
        c0=args.c0,
        lambda_s=args.lambda_s,
        trust_power=args.trust_power,
        pi_min=args.pi_min,
        pi_max=args.pi_max,
        schur_block=args.schur_block,
    )
    samples_norm_tj3, samples_world = dit_norm_to_world(
        samples_norm,
        mean,
        std,
        n_joints=args.raw_joint_count,
    )
    save_raw_inpaint_outputs(
        args,
        samples_norm_tj3=samples_norm_tj3,
        samples_world=samples_world,
        control_world=control_world,
        control_mask_chw=mask_chw,
        trajectory_requested=trajectory_requested,
        trajectory_used=trajectory_used,
        joint_ids=joint_ids,
        test_motion_id=test_motion_id,
        test_prompt_index=test_prompt_index,
        sparse_control_metadata=sparse_control_metadata,
    )


if __name__ == "__main__":
    main()
