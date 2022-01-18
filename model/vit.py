import copy
import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from timm.models.helpers import named_apply
from timm.models.layers.weight_init import lecun_normal_, trunc_normal_

from model.model_utils.vit_helpers import _load_weights


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def traid(t):
    return t if isinstance(t, tuple) else (t, t, t)

# We have to adapt the different layers for 3D and try to align this with the `timm` implementations
class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, volume_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        volume_size = traid(volume_size)
        patch_size = traid(patch_size)
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.grid_size = (volume_size[0] // patch_size[0], volume_size[1] // patch_size[1], volume_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L, H, W = x.shape
        assert L == self.volume_size[0] and H == self.volume_size[1] and W == self.volume_size[2], \
            f"Volume image size ({L}*{H}*{W}) doesn't match model ({self.volume_size[0]}*{self.volume_size[1]}*{self.volume_size[2]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Mlp3D(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# The Attention is always applied to the sequences. Thus, at this point, it should be the same model
# whether we apply it in NLP, ViT, speech or any other domain :)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp3D(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer3D(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, volume_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed3D, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            volume_size (int, triad): input image size
            patch_size (int, triad): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            volume_size=volume_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x





# class Embeddings(nn.Module):
#     """Construct the embeddings from patch, position embeddings.
#     """
#
#     def __init__(self, img_size, in_channels, hidden_size=252, down_factor=2, patch_size=(8, 8, 8), dropout_embed=0.1):
#         super(Embeddings, self).__init__()
#         n_patches = int((img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2]// patch_size[2])) + 1 # One more for CLS token
#         self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
#                                           out_channels=hidden_size,
#                                           kernel_size=patch_size,
#                                           stride=patch_size)
#         self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
#         self.dropout = nn.Dropout(dropout_embed)
#
#     def forward(self, x):
#         x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
#         x = x.flatten(2)
#         x = x.transpose(-1, -2)  # (B, n_patches, hidden)
#         b, n, _ = x.shape
#
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         embeddings = x + self.position_embeddings
#         embeddings = self.dropout(embeddings)
#
#         return embeddings


# class Block(nn.Module):
#     def __init__(self, hidden_size=252, vis=False, mlp_dim=3072, mlp_dropout=0.1, num_heads=12, dropout_attn=0.):
#         super(Block, self).__init__()
#         self.hidden_size = hidden_size
#         self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
#         self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
#         self.ffn = Mlp(hidden_size=hidden_size, mlp_dim=mlp_dim, mlp_dropout=mlp_dropout)
#         self.attn = Attention(num_heads, hidden_size=hidden_size, dropout_p=dropout_attn, vis=vis)
#
#     def forward(self, x):
#         h = x
#
#         x = self.attention_norm(x)
#         x, weights = self.attn(x)
#         x = x + h
#
#         h = x
#         x = self.ffn_norm(x)
#         x = self.ffn(x)
#         x = x + h
#         return x, weights


# class Encoder(nn.Module):
#     def __init__(self, hidden_size, num_layers=12, vis=False, mlp_dim=3072, mlp_dropout=0.1, num_heads=12,
#                  dropout_attn=0.):
#         super(Encoder, self).__init__()
#         self.vis = vis
#         self.layer = nn.ModuleList()
#         self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
#         for _ in range(num_layers):
#             layer = Block(hidden_size=hidden_size, vis=vis, mlp_dim=mlp_dim, mlp_dropout=mlp_dropout,
#                           num_heads=num_heads, dropout_attn=dropout_attn)
#             self.layer.append(copy.deepcopy(layer))
#
#     def forward(self, hidden_states):
#         attn_weights = []
#         for layer_block in self.layer:
#             hidden_states, weights = layer_block(hidden_states)
#             if self.vis:
#                 attn_weights.append(weights)
#         encoded = self.encoder_norm(hidden_states)
#         return encoded, attn_weights


# class Transformer(nn.Module):
#     def __init__(self, img_size, in_channels, hidden_size=252, down_factor=2, patch_size=(8, 8, 8), mlp_dropout=0.1,
#                  num_layers=12, mlp_dim=3072, num_heads=12, dropout_embed=0., dropout_attn=0., vis=False):
#         super(Transformer, self).__init__()
#         self.embeddings = Embeddings(img_size=img_size, in_channels=in_channels, hidden_size=hidden_size,
#                                      down_factor=down_factor, patch_size=patch_size, dropout_embed=dropout_embed)
#         self.encoder = Encoder(hidden_size=hidden_size, num_layers=num_layers, mlp_dim=mlp_dim, mlp_dropout=mlp_dropout,
#                                num_heads=num_heads,
#                                dropout_attn=dropout_attn, vis=vis)
#
#     def forward(self, input_ids):
#         embedding_output = self.embeddings(input_ids)
#         encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
#         return encoded, attn_weights


# class Conv3dReLU(nn.Sequential):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size,
#             padding=0,
#             stride=1,
#             use_batchnorm=True,
#     ):
#         conv = nn.Conv3d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=not (use_batchnorm),
#         )
#         relu = nn.ReLU(inplace=True)
#
#         bn = nn.BatchNorm3d(out_channels)
#
#         super(Conv3dReLU, self).__init__(conv, bn, relu)


# def traid(t):
#     return t if isinstance(t, tuple) else (t, t, t)


# class ViT(nn.Module):
#     def __init__(self, image_size, num_classes, in_channels, pool='cls', channels=3, hidden_size=252, down_factor=2,
#                  patch_size=(8, 8, 8), mlp_dropout=0.1,
#                  num_layers=12, mlp_dim=3072, num_heads=12, dropout_embed=0., dropout_attn=0., vis=False):
#         super().__init__()
#         image_height, image_width, image_depth = traid(image_size)
#         patch_height, patch_width, patch_depth = traid(patch_size)
#
#         assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth % patch_depth == 0, 'Image dimensions must be divisible by the patch size.'
#
#         num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
#         patch_dim = channels * patch_height * patch_width * patch_depth
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#
#         self.transformer = Transformer(img_size=image_size, in_channels=in_channels, hidden_size=hidden_size,
#                                        down_factor=down_factor, patch_size=patch_size, mlp_dropout=mlp_dropout,
#                                        num_layers=num_layers, mlp_dim=mlp_dim, num_heads=num_heads,
#                                        dropout_embed=dropout_embed, dropout_attn=dropout_attn, vis=vis)
#
#         self.pool = pool
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(hidden_size),
#             nn.Linear(hidden_size, num_classes)
#         )
#
#     def forward(self, img):
#         x, attn_wt = self.transformer(img)
#         x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
#         return self.mlp_head(x)


if __name__ == '__main__':
    image_size = (64, 64, 64)
    sample_img = torch.randn(8, 3, 64, 64, 64)
    # model = ViT(image_size=image_size, num_classes=2, in_channels=5)
    # # Put things to cuda and check
    # model.cuda()
    # sample_img = sample_img.cuda()
    # output = model(sample_img)
    # print(output.shape)
    embed = VisionTransformer3D(volume_size=image_size, in_chans=3)
    output = embed(sample_img)
    print(output.shape)

