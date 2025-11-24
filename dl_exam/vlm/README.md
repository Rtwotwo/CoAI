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

