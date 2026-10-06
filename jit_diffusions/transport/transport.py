import os
import torch as th
import numpy as np
import logging

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        prediction_output="velocity",
        x_t_sampler="logit",
        logit_mu=-0.8,
        sigma_min=0.05,
        num_frame_per_block=3,
        is_causal=True,
        t_per_frame=False,
        **kwargs,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        if sigma_min < 0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}")
        self.sigma_min = sigma_min
        valid_outputs = {"velocity", "x"}
        if prediction_output not in valid_outputs:
            raise ValueError(
                f"prediction_output must be one of {valid_outputs}, got {prediction_output}"
            )
        if self.model_type != ModelType.VELOCITY and prediction_output != "velocity":
            raise ValueError(
                "prediction_output='x' is only supported when model_type is VELOCITY"
            )
        self.prediction_output = prediction_output
        valid_samplers = {"logit", "uniform"}
        if x_t_sampler not in valid_samplers:
            raise ValueError(
                f"x_t_sampler must be one of {valid_samplers}, got {x_t_sampler}"
            )
        self.x_t_sampler = x_t_sampler
        self.logit_mu = float(logit_mu)
        self.num_frame_per_block = num_frame_per_block
        self.is_causal = is_causal
        self.t_per_frame = t_per_frame

    def prior_logp(self, z):
        """
        Standard multivariate normal prior
        Assume z is batched
        """
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0
        return th.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):  # avoid numerical issue by taking a first semi-implicit step

            t0 = (
                eps
                if (diffusion_form == "SBDM" and sde)
                or self.model_type != ModelType.VELOCITY
                else 0
            )
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
        """

        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if self.model_type == ModelType.VELOCITY and self.prediction_output == "x":
            if self.x_t_sampler == "logit":
                # logit-normal mapped to [t0, t1]
                mu, sigma = self.logit_mu, 0.8
                z = (
                    th.randn(
                        (x1.shape[0], x1.shape[1] if (self.is_causal or self.t_per_frame) else 1),
                        device=x1.device,
                    )
                    * sigma
                    + mu
                )
                t = th.sigmoid(z)
                t = t0 + (t1 - t0) * t
                t = th.clamp(t, min=t0, max=t1)
            else:  # uniform
                t = (
                    th.rand(
                        (x1.shape[0], x1.shape[1] if (self.is_causal or self.t_per_frame) else 1),
                        device=x1.device,
                    )
                    * (t1 - t0)
                    + t0
                )
        else:
            t = (
                th.rand(
                    (x1.shape[0], x1.shape[1] if (self.is_causal or self.t_per_frame) else 1),
                    device=x1.device,
                )
                * (t1 - t0)
                + t0
            )
        if self.is_causal:
            t = t.reshape(t.shape[0], -1, self.num_frame_per_block)
            t[:, :, 1:] = t[:, :, 0:1]
            t = t.reshape(t.shape[0], -1)
        elif not self.t_per_frame:
            t = t.expand(-1, x1.shape[1])
        return t, x0, x1

    def training_losses(self, model, x1, model_kwargs=None, ae=None, dim=(1, 3)):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}

        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        if (self.model_type == ModelType.VELOCITY) and (ae is not None):
            gt = model_kwargs["control"].permute(0, 2, 3, 1).clone()
            idx = model_kwargs["index"].clone()
            after_mean, after_std, train_mean, train_std = model_kwargs["mean_std"]
            del model_kwargs["mean_std"]
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {}
        terms["pred"] = model_output
        if self.model_type == ModelType.VELOCITY:
            pred_velocity = self._convert_output_to_velocity(xt, t, model_output)
            if self.prediction_output == "x":
                sigma_t, _ = self.path_sampler.compute_sigma_t(
                    path.expand_t_like_x(t, xt)
                )
                denom = th.clamp(sigma_t, min=self.sigma_min)
                v_target = (x1 - xt) / denom
            else:
                v_target = ut

            terms["loss"] = mean_flat((pred_velocity - v_target) ** 2, dim=dim)

            if ae is not None:
                for param in ae.parameters():
                    param.requires_grad = False
                after_mean = th.from_numpy(after_mean).to(model_output.device)
                after_std = th.from_numpy(after_std).to(model_output.device)
                train_mean = th.from_numpy(train_mean).to(model_output.device)
                train_std = th.from_numpy(train_std).to(model_output.device)
                if self.prediction_output == "x":
                    pred_x1 = model_output
                else:
                    pred_x1 = x0 + pred_velocity
                pred = pred_x1.permute(0, 2, 3, 1)
                pred = (pred * after_std) + after_mean
                pred = ae.decode(pred.permute(0, 3, 1, 2))
                pred = pred * train_std.clone() + train_mean.clone()
                gt = gt * train_std.clone() + train_mean.clone()
                terms["loss_control"] = (
                    ((pred - gt) ** 2) * (idx.permute(0, 2, 3, 1))
                ).sum() / idx.sum()
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t**2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms["loss"] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms["loss"] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        return terms

    def get_drift(self):
        """member function for obtaining the drift of the probability flow ODE"""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output  # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            model_velocity = self._convert_output_to_velocity(x, t, model_output)
            return model_velocity

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert (
                model_output.shape == x.shape
            ), "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(
        self,
    ):
        """member function for obtaining score of
        x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = (
                lambda x, t, model, **kwargs: model(x, t, **kwargs)
                / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = (
                lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(
                    self._convert_output_to_velocity(x, t, model(x, t, **kwargs)), x, t
                )
            )
        else:
            raise NotImplementedError()

        return score_fn

    def _convert_output_to_velocity(self, xt, t, model_output):
        if self.model_type != ModelType.VELOCITY:
            return model_output

        if self.prediction_output == "velocity":
            return model_output

        t_like = path.expand_t_like_x(t, xt)
        sigma_t, _ = self.path_sampler.compute_sigma_t(t_like)
        denom = th.clamp(sigma_t, min=self.sigma_min)
        return (model_output - xt) / denom


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):
        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )
            return diffusion

        sde_drift = lambda x, t, model, **kwargs: self.drift(
            x, t, model, **kwargs
        ) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)

        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x
        elif last_step == "Mean":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x
                + sde_drift(x, t, model, **model_kwargs) * last_step_size
            )
        elif last_step == "Tweedie":
            alpha = (
                self.transport.path_sampler.compute_alpha_t
            )  # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = lambda x, t, model, **model_kwargs: x / alpha(t)[0][0] + (
                th.clamp(sigma(t)[0][0], min=self.transport.sigma_min) ** 2
            ) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x
                + self.drift(x, t, model, **model_kwargs) * last_step_size
            )
        else:
            raise NotImplementedError()

        return last_step_fn

    def _heun2_with_last_euler(
        self, init, model, drift_fn, t0, t1, num_steps, **model_kwargs
    ):
        """
        Fixed-step Heun2 (trapezoidal) following the JiT-style schedule:
          - Heun2 for all but the last step
          - Final step is a single Euler step
        Returns a tensor of shape [num_steps, *init.shape].
        """
        device = init.device
        batch = init.shape[0]
        t_grid = th.linspace(t0, t1, num_steps, device=device)

        x = init
        xs = [x]
        # Heun2 for intermediate steps (up to the penultimate time)
        for i in range(num_steps - 2):
            t_i = t_grid[i]
            t_next = t_grid[i + 1]
            dt = t_next - t_i

            t_vec = th.full((batch,), t_i, device=device)
            t_next_vec = th.full((batch,), t_next, device=device)

            # k1
            v_t = drift_fn(x, t_vec, model, **model_kwargs)

            # Euler predictor
            x_euler = x + dt * v_t

            # k2
            v_t_next = drift_fn(x_euler, t_next_vec, model, **model_kwargs)

            # Trapezoidal update
            v = 0.5 * (v_t + v_t_next)
            x = x + dt * v
            xs.append(x)

        # Final step: Euler only
        if num_steps >= 2:
            t_i = t_grid[-2]
            t_next = t_grid[-1]
            dt = t_next - t_i

            t_vec = th.full((batch,), t_i, device=device)
            v_t = drift_fn(x, t_vec, model, **model_kwargs)
            x = x + dt * v_t
            xs.append(x)

        return th.stack(xs, dim=0)

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self.__get_last_step(
            sde_drift, last_step=last_step, last_step_size=last_step_size
        )

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="myheun2",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(
                x, th.ones_like(t) * (1 - t), model, **kwargs
            )
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        if sampling_method == "myheun2":
            # Use fixed-step Heun2 with final Euler step (JiT-style) instead of torchdiffeq integrators.
            def _sample_fn(init, model, **model_kwargs):
                xs = self._heun2_with_last_euler(
                    init, model, drift, t0, t1, num_steps, **model_kwargs
                )
                return xs

            return _sample_fn
        else:
            return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):

        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """

        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(
                    th.sum(self.drift(x, t, model, **model_kwargs) * eps), x
                )[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
