# :rocket:Model Introduction:rocket:

## :house:1.CLIP_Origin Architecture:house:

When constructing the CLIP model, the code manually builds the dependency relationships of each component through a layered and modular design: first, basic components such as Bottleneck (for ModifiedResNet), AttentionPool2d (implementing attention pooling), and ResidualAttentionBlock (the basic unit of Transformer) are defined. Then, based on these, two types of visual encoders are constructed respectively: ModifiedResNet (suitable for the CNN backbone) and VisionTransformer (suitable for the ViT backbone). At the same time, a Transformer encoder for the text side is built, and both are uniformly integrated into the main CLIP class. CLIP automatically selects the visual backbone according to the type of vision\_layers, aligns the image and text feature spaces through the shared embed\_dim, and adjusts the similarity output using the learnable logit\_scale. The entire dependency chain is bottom-up, from underlying convolution/attention operations to high-level multimodal alignment logic, with layers of encapsulation and parameter collaboration, ultimately forming an end-to-end contrastive learning architecture. The specific code can be viewed in detail in [clip.py](dl_exam/vlm/clip_origin/model.py), and this code refers to and learns from the [CLIP](https://github.com/openai/CLIP.git) repository.

```mermaid
classDiagram

    class Bottleneck {
        +expansion: int = 4
        +__init__(inplanes, planes, stride)
        +forward(x)
    }

    class AttentionPool2d {
        +__init__(spacial_dim, embed_dim, num_heads, output_dim)
        +forward(x)}

    class ModifiedResNet {
        +__init__(layers, output_dim, heads, input_resolution, width)
        +forward(x)
    }

    class LayerNorm {
        +forward(x)
    }

    class QuickGELU {
        +forward(x)
    }

    class ResidualAttentionBlock {
        +__init__(d_model, n_head, attn_mask)
        +forward(x)
    }

    class Transformer {
        +__init__(width, layers, heads, attn_mask)
        +forward(x)
    }

    class VisionTransformer {
        +__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        +forward(x)
    }

    class CLIP {
        +__init__(...)
        +encode_image(image)
        +encode_text(text)
        +forward(image, text)
        .. visual: ModifiedResNet or VisionTransformer ..
    }

    %% 组合关系（仅自定义类之间）

    ModifiedResNet *-- "1..*" Bottleneck : layers
    ModifiedResNet *-- "1" AttentionPool2d : attnpool

    VisionTransformer *-- "1" Transformer : transformer
    VisionTransformer *-- "2" LayerNorm : ln_pre, ln_post

    Transformer *-- "1..*" ResidualAttentionBlock : resblocks

    ResidualAttentionBlock *-- "2" LayerNorm : ln_1, ln_2
    ResidualAttentionBlock *-- "1" QuickGELU : in mlp
    CLIP *-- "1" ModifiedResNet : visual (ResNet)
    CLIP *-- "1" VisionTransformer : visual (ViT)
    CLIP *-- "1" Transformer : text transformer
    CLIP *-- "1" LayerNorm : ln_final
```

## :house:2.CLIP_Latest Architecture:house:

The construction process of [MultimodalTransformer](dl_exam/vlm/clip_latest/transformer.py) takes the standard Transformer encoder as its core framework, and realizes cross-modal interaction through multi-layer stacked ResidualAttentionBlocks. Each ResidualAttentionBlock contains a custom Attention module (supporting cross-attention mechanism for fusing image and text features), a feed-forward network MLP driven by the [QuickGELU](dl_exam/base/utils/activation.py) activation function, and an optional high-precision normalization layer LayerNormFp32 to stabilize the training process. The entire Transformer body is encapsulated in the MultimodalTransformer class, which is responsible for receiving embedding sequences from the visual encoder and text encoder. Under the iterative action of multi-layer residual attention blocks, it gradually aligns and fuses the semantic information of the two modalities, and finally outputs a joint multimodal representation, thereby realizing the modeling of deep semantic association between images and texts.

```mermaid
classDiagram

    %% ==================== 核心类 ====================
    class MultimodalTransformer {
        +__init__(width, layers, heads, context_length, mlp_ratio, ls_init value, act_layer, norm_layer, output_dim, batch_first)
        +forward(img_embs, text_embs) Tensor
    }

    class Transformer {
        +__init__(width, layers, heads, mlp_ratio, ls_init_value, act_layer, norm_layer, batch_first, block_type)
        +forward(x, attn_mask) Tensor
        +forward_intermediates(...) Tuple[Tensor, List[Tensor]]
    }

    class ResidualAttentionBlock {
        +__init__(d_model, n_head, mlp_ratio, ls_init_value, act_layer, norm_layer, is_cross_attention, batch_first)
        +forward(q_x, k_x, v_x, attn_mask) Tensor
    }

    class Attention {
        +__init__(dim, num_heads, qkv_bias, qk_norm, scaled_cosine, scale_heads, inner_norm, norm_layer, attn_drop, proj_drop)
        +forward(x: Tensor, attn_mask) Tensor
    }

    class LayerNormFp32 {
        +forward(x: Tensor) Tensor
    }

    class QuickGELU {
        +forward(x: Tensor) Tensor
    }

    class MLP {
        +__init__(in_features, hidden_features, out_features, act_layer, drop)
        +forward(x: Tensor) Tensor
    }

    %% ==================== 组合关系 ====================
    MultimodalTransformer "1" *-- "1" Transformer : contains
    Transformer "1" *-- "layers" ResidualAttentionBlock : contains
    ResidualAttentionBlock "1" *-- "1" Attention : uses
    ResidualAttentionBlock "1" *-- "1" MLP : uses
    ResidualAttentionBlock "1" *-- "1" LayerNormFp32 : optional ln_q/ln_k
    ResidualAttentionBlock "1" *-- "1" QuickGELU : in MLP
    Attention "1" *-- "1" LayerNormFp32 : for query/key normalization (optional)
    Attention "1" *-- "1" QuickGELU : in attention projection
```
