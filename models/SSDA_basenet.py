from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from timm.models.registry import register_model
from timm import create_model
from timm.models.layers import Mlp, PatchEmbed, DropPath
from timm.models.vision_transformer import VisionTransformer
from collections import OrderedDict
from functools import partial
from einops import rearrange


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


grad_reverse = RevGrad.apply


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True, bootleneck_dim=256):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features
        self.bottle_neck = feat_bootleneck(feature_dim=4096, bottleneck_dim=bootleneck_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x=self.bottle_neck(x)
        return x

    def output_num(self):
        return self.__in_feature


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or \
       classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)


class MLP(nn.Module):
    def __init__(self, pret=True,input_dim=3000,bootleneck_dim=256):
        super(MLP, self).__init__()
        self.features = nn.Sequential(nn.Linear(input_dim,out_features=500))
        self.features.apply(init_weights)
        self.__in_features = 500
        self.bottle_neck = feat_bootleneck(feature_dim=500, bottleneck_dim=bootleneck_dim)
        self.bottle_neck.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        x=self.bottle_neck(x)
        return x

    def output_num(self):
        return self.__in_feature


class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="bn"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        x = self.dropout(x)
        return x


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False,bootleneck_dim=256):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.bottle_neck=feat_bootleneck(feature_dim=4096,bottleneck_dim=bootleneck_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        x = self.bottle_neck(x)
        return x


class VGGBase_no_neck(nn.Module):
    def __init__(self, pret=True, no_pool=False,bootleneck_dim=256):
        super(VGGBase_no_neck, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05, norm_feature=1):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=True)
        self.fc.apply(init_weights)
        self.num_class = num_class
        self.temp = temp
        self.norm_feature = norm_feature
    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, torch.tensor(eta, requires_grad=False))
        if self.norm_feature:
            x = F.normalize(x)
            x_out = self.fc(x) / self.temp
        else:
            x_out = self.fc(x)
        return x_out


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, norm_feature=1,temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, inc//2)
        self.fc1.apply(init_weights)
        self.fc2 = nn.Linear(inc//2, num_class,bias=False)
        nn.init.xavier_normal_(self.fc2.weight)
        self.bn = nn.BatchNorm1d(inc//2, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.num_class = num_class
        self.temp = temp
        self.norm_feature=norm_feature

    def forward(self, x, reverse=False, eta=0.1):
        x = self.dropout(self.relu(self.bn(self.fc1(x))))
        if self.norm_feature:
            x = F.normalize(x)
            x_out = self.fc2(x) / self.temp
        else:
            x_out = self.fc2(x)
        return x_out


class ViT_timm(nn.Module):
    def __init__(self, bootleneck_dim=256):
        super(ViT_timm, self).__init__()
        self.backbone = create_model('ds_deit_small_patch16_224', pretrained=True)
        self.in_features = self.backbone.num_features
        self.num_patch = self.backbone.patch_embed.num_patches
        self.bottle_neck = feat_bootleneck(feature_dim = self.in_features, bottleneck_dim = bootleneck_dim)

    def forward(self, x):
        x_token = self.backbone.patch_embed(x)
        x_logits, x_p, x_attn = self.backbone.forward_features(x_token, patch=True)
        x_logits = self.bottle_neck(x_logits)
        return x_logits, x_p, x_attn, x_token


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
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        save = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, save
    

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        t, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(t)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn    
    

class Vit(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
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
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
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

    def forward_features(self, x, patch=False):
        if not patch:
            x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        attns = []
        for b in self.blocks:
            x, attn = b(x)
            if self.dist_token is not None:
                attns.append(attn[:,:,0,2:])
            else:
                attns.append(attn[:,:,0,1:])
        attns = torch.mean(torch.stack(attns, dim=0), dim=2)# avg head
        attns = torch.mean(attns, dim=0)# avg layer
        x = self.norm(x)
        if self.dist_token is not None:
            return x[:, 0], x[:, 2:], attns
        else:
            return x[:, 0], x[:, 1:], attns


@register_model
def ds_deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, distilled=True, **kwargs)
    model = Vit(**model_kwargs)
    if pretrained:
        pre = create_model('deit_small_distilled_patch16_224', pretrained=pretrained)
        model.load_state_dict(pre.state_dict())
    return model