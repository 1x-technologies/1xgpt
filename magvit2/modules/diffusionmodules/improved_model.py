import torch
import torch.nn as nn

from magvit2.config import VQConfig


def swish(x):
    # swish
    return x*torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(self, 
                 in_filters,
                 out_filters,
                 use_conv_shortcut = False
                 ) -> None:
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x, **kwargs):
        residual = x

        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual


class Encoder(nn.Module):
    def __init__(
        # self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4),
        self, config: VQConfig,
    ):
        super().__init__()

        self.in_channels = config.in_channels
        self.z_channels = config.z_channels

        self.num_res_blocks = config.num_res_blocks
        self.num_blocks = len(config.ch_mult)
        
        self.conv_in = nn.Conv2d(
            config.in_channels,
            config.base_channels,
            kernel_size=(3, 3),
            padding=1,
            bias=False
        )

        ## construct the model
        self.down = nn.ModuleList()

        in_ch_mult = (1,) + tuple(config.ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = config.base_channels * in_ch_mult[i_level]  # [1, 1, 2, 2, 4]
            block_out = config.base_channels * config.ch_mult[i_level]  # [1, 2, 2, 4]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(block_out, block_out, kernel_size=(3, 3), stride=(2, 2), padding=1)

            self.down.append(down)
        
        ### mid
        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        
        ### end
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, config.z_channels, kernel_size=(1, 1))
            
    def forward(self, x):
        ## down
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
            
            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)
        
        ## mid 
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    # def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4)) -> None:
    def __init__(self, config: VQConfig) -> None:
        super().__init__()

        self.base_channels = config.base_channels
        self.num_blocks = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks

        block_in = self.base_channels * config.ch_mult[self.num_blocks - 1]

        self.conv_in = nn.Conv2d(
            config.z_channels, block_in, kernel_size=(3, 3), padding=1, bias=True
        )

        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        
        self.up = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = self.base_channels * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)
        
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)

        self.conv_out = nn.Conv2d(block_in, config.out_channels, kernel_size=(3, 3), padding=1)
    
    def forward(self, z):

        z = self.conv_in(z)

        ## mid
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)
        
        ## upsample
        for i_level in reversed(range(self.num_blocks)):
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)
            
            if i_level > 0:
                z = self.up[i_level].upsample(z)
        
        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)

        return z


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size,
                            w * block_size)

    return x


class Upsampler(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out
        

# if __name__ == "__main__":
#     x = torch.randn(size = (2, 3, 128, 128))
#     encoder = Encoder(ch=128, in_channels=3, num_res_blocks=2, z_channels=18, out_ch=3, resolution=128)
#     decoder = Decoder(out_ch=3, z_channels=18, num_res_blocks=2, ch=128, in_channels=3, resolution=128)
#     z = encoder(x)
#     out = decoder(z)
