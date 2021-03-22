import torch
from nflows import transforms, flows
from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.transforms.made import MaskedLinear, _get_input_degrees, MaskedResidualBlock, MaskedFeedforwardBlock
from sbi.utils import standardizing_transform, standardizing_net, torchutils
from torch import nn, Tensor, tanh
import torch.nn.functional as F
from pyknos.nflows import distributions as distributions_


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,2,2)),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64, out_channels=50, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.AdaptiveAvgPool3d((1,1,1))
        )

    def forward(self, x):
        x = x.view(-1,1,128,128,128)
        return self.net(x).squeeze()

class DE(nn.Module):
    """Implementation of MADE.

    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        output_multiplier=1,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__()

        # Initial layer.
        self.initial_layer = MaskedLinear(
            in_degrees=_get_input_degrees(features),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False,
        )

        if context_features is not None:
            self.context_layer = ConvNet()

        self.use_residual_blocks = use_residual_blocks
        self.activation = activation
        # Residual blocks.
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedforwardBlock
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=context_features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True,
        )

    def forward(self, inputs, context=None):
        temps = self.initial_layer(inputs)
        if context is not None:
            temps += self.activation(self.context_layer(context))
        if not self.use_residual_blocks:
            temps = self.activation(temps)
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        return outputs

class MaskedAffineAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        self.features = features
        made = DE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self._epsilon = 1e-3
        super(MaskedAffineAutoregressiveTransform, self).__init__(made)

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._epsilon
        scale = F.softplus(unconstrained_scale) + self._epsilon
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._epsilon
        scale = F.softplus(unconstrained_scale) + self._epsilon
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        return unconstrained_scale, shift


def build_tumorde(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> nn.Module:
    """Builds MAF p(x|y).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        z_score_y: Whether to z-score ys passing into the network.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    y_numel = embedding_net(batch_y[:1]).numel()

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    MaskedAffineAutoregressiveTransform(
                        features=x_numel,
                        hidden_features=hidden_features,
                        context_features=y_numel,
                        num_blocks=2,
                        use_residual_blocks=False,
                        random_mask=False,
                        activation=tanh,
                        dropout_probability=0.0,
                        use_batch_norm=True,
                    ),
                    transforms.RandomPermutation(features=x_numel),
                ]
            )
            for _ in range(num_transforms)
        ]
    )

    if z_score_x:
        transform_zx = standardizing_transform(batch_x)
        transform = transforms.CompositeTransform([transform_zx, transform]).cuda()

    if z_score_y:
        embedding_net = nn.Sequential(standardizing_net(batch_y), embedding_net)

    distribution = distributions_.StandardNormal((x_numel,)).cuda()
    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net