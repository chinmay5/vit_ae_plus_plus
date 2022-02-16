from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from model.model_utils.gaussian_filter import perform_3d_gaussian_blur
from model.model_utils.perceptual_loss import vgg_perceptual_loss
from model.model_utils.sobel_filter import SobelFilter3d
from model.model_utils.vit_helpers import get_3d_sincos_pos_embed
from model.vit import PatchEmbed3D, Block


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, volume_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, args=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed3D(volume_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 3 * in_chans, bias=True)  # encoder to decoder
        self.sobel_filter3D = SobelFilter3d()
        self.perceptual_loss = vgg_perceptual_loss()
        self.args = args
        self.perceptual_weight = 1 if self.args is None else self.args.perceptual_weight
        print(f"Using perceptual weight of {self.perceptual_weight}")

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], round(self.patch_embed.num_patches ** (1 / 3)),
                                            # int(self.patch_embed.num_patches ** (1/3)),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.patch_embed.num_patches ** .5), cls_token=True)
        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    round(self.patch_embed.num_patches ** (1 / 3)), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, volume):
        """
        volume: (N, 3, L, H, W)
        x: (N, L, patch_size**3 *3)
        """
        p = self.patch_embed.patch_size[0]  # Patch size
        assert volume.shape[2] == volume.shape[3] == volume.shape[4] and volume.shape[
            2] % p == 0  # Ensuring we have the same dimension

        l = h = w = volume.shape[2] // p  # Since volumes have the same dimension. Possible limitation??
        x = volume.reshape(shape=(volume.shape[0], -1, l, p, h, p, w, p))
        x = torch.einsum('nclrhpwq->nlhwrpqc', x)
        x = x.reshape(shape=(volume.shape[0], l * h * w, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        l = h = w = round(x.shape[1] ** (1 / 3))
        assert l * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], l, h, w, p, p, p,
                             -1))  # Earlier 3 was hard-coded here. Maybe this way, we are more flexible with the number of channels
        x = torch.einsum('nlhwrpqc->nclrhpwq', x)
        volume = x.reshape(shape=(x.shape[0], -1, h * p, h * p, h * p))
        return volume

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, edge_map_weight=0):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = self.get_weighted_loss(pred, target, mask, edge_map_weight=edge_map_weight)
        return loss

    def get_weighted_loss(self, pred, target, mask, edge_map_weight=0):
        pred_vol, target_vol = self.unpatchify(pred), self.unpatchify(target)
        pred_edge_map, orig_input_edge_map = self.sobel_filter3D(pred_vol), self.sobel_filter3D(
            perform_3d_gaussian_blur(target_vol, blur_sigma=2))
        edge_map_loss = edge_map_weight * F.mse_loss(pred_edge_map, orig_input_edge_map, reduction="mean")
        reconstruction_loss = ((pred - target) ** 2).mean(dim=-1)
        reconstruction_loss = (reconstruction_loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # Including the perceptual loss
        with torch.no_grad():
            percep_loss = self.perceptual_weight * self.perceptual_loss(pred_vol, target_vol)
        loss = edge_map_loss + reconstruction_loss + percep_loss
        return [loss, edge_map_loss, reconstruction_loss, percep_loss]

    def forward(self, sample, mask_ratio=0.75, edge_map_weight=0):
        latent, mask, ids_restore = self.forward_encoder(sample, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(sample, pred, mask, edge_map_weight=edge_map_weight)
        return loss, pred, mask


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks

if __name__ == '__main__':
    image_size = (96, 96, 96)
    model = mae_vit_base_patch16(volume_size=image_size, in_chans=1, patch_size=8)
    sample_img = torch.randn(8, 1, 96, 96, 96)
    loss, pred, mask = model(sample_img)
    print(pred.shape)
    print(loss[0].item())
    print(loss[1].item())
    print(loss[2].item())
    print(loss[3].item())
