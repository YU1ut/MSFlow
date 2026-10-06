from .transport import Transport, ModelType, WeightType, PathType, Sampler


def create_transport(
    path_type="Linear",
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    prediction_output="velocity",
    x_t_sampler="logit",
    logit_mu=-0.8,
    sigma_min=0.05,
    num_frame_per_block=3,
    is_causal=False,
    t_per_frame=False,
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weight loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    - prediction_output: when model_type is velocity, choose whether the network outputs
      the velocity directly ("velocity") or predicts x and the loss is applied on the
      induced velocity ("x")
    - x_t_sampler: strategy to sample t when prediction_output=='x' (only training time)
    - logit_mu: mean of the normal before sigmoid when x_t_sampler=='logit'
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path_type = path_choice[path_type]

    if path_type in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif (
        path_type in [PathType.GVP, PathType.LINEAR]
        and model_type != ModelType.VELOCITY
    ):
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-2 if train_eps is None else sample_eps
    else:  # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0

    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        prediction_output=prediction_output,
        x_t_sampler=x_t_sampler,
        logit_mu=logit_mu,
        sigma_min=sigma_min,
        num_frame_per_block=num_frame_per_block,
        is_causal=is_causal,
        t_per_frame=t_per_frame,
    )

    return state
