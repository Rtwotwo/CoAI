# :rocket:Model Introduction:rocket:

Vision-Language Models (VLMs), as the core carrier of multimodal artificial intelligence, have made significant breakthroughs in recent years in terms of architectural design, training paradigms, and application scenarios. Modern VLMs are based on Transformers as their backbone, establishing a cross-modal semantic foundation through large-scale image-text alignment pre-training (such as CLIP and ALIGN), and have gradually evolved into general multimodal bases that support fine-grained understanding, generation, and reasoning. Current mainstream VLM architectures generally adopt a "two-tower" or "fusion encoder" structure. During the pre-training phase, they combine contrastive learning, mask modeling, and generative objectives to achieve deep alignment between images and text. In the post-alignment phase, they enhance the model's generalization and controllability in open-world tasks through instruction tuning, reinforcement learning from human feedback (RLHF), and domain adaptation strategies. Cutting-edge VLMs not only support traditional tasks such as image-text retrieval, Visual Question Answering (VQA), and image captioning but also demonstrate strong zero-shot transfer capabilities, multi-step reasoning abilities (such as Chain-of-Thought in VLM), as well as the potential for tool invocation and interaction with external knowledge. Representative models include: BLIP/BLIP-2 (efficient modular design), Flamingo (few-shot learning based on frozen pre-trained models), KOSMOS-1/2 (unified multimodal sequence modeling), LLaVA/LLaVA-NeXT (open-source multimodal dialogue agents), Qwen-VL/Qwen2-VL (supporting high-resolution and complex layout understanding), IDEFICS2 (open science-oriented multilingual VLM), as well as closed-source system-level models such as Google’s PaLI-X, Gemini series, and OpenAI’s GPT-4V(ision).

## :house:1.CLIP_Origin Architecture:house:

When constructing the CLIP model, the code manually builds the dependency relationships of each component through a layered and modular design: first, basic components such as Bottleneck (for ModifiedResNet), AttentionPool2d (implementing attention pooling), and ResidualAttentionBlock (the basic unit of Transformer) are defined. Then, based on these, two types of visual encoders are constructed respectively: ModifiedResNet (suitable for the CNN backbone) and VisionTransformer (suitable for the ViT backbone). At the same time, a Transformer encoder for the text side is built, and both are uniformly integrated into the main CLIP class. CLIP automatically selects the visual backbone according to the type of vision\_layers, aligns the image and text feature spaces through the shared embed\_dim, and adjusts the similarity output using the learnable logit\_scale. The entire dependency chain is bottom-up, from underlying convolution/attention operations to high-level multimodal alignment logic, with layers of encapsulation and parameter collaboration, ultimately forming an end-to-end contrastive learning architecture. The specific code can be viewed in detail in [clip.py](https://github.com/Rtwotwo/Code-Exam/blob/main/dl_exam/vlm/clip_origin/clip.py), and this code refers to and learns from the [CLIP](https://github.com/openai/CLIP.git) repository.

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

```mermaid
flowchart LR
    subgraph Image Tower [Vision Transformer]
        A["Input Image\n[B, 3, 224, 224]"] --> B["Patch Embedding\n(Conv2d, patch=16)\n→ [B, 768, 14, 14]"]
        B --> C["Reshape → [B, 768, 196]"]
        C --> D["Permute → [B, 196, 768]"]
        D --> E["Add Class Token\n→ [B, 197, 768]"]
        E --> F["Add Pos Emb\n→ [B, 197, 768]"]
        F --> G["LN Pre\n→ [B, 197, 768]"]
        G --> H["Transformer ×12\n→ [B, 197, 768]"]
        H --> I["Take cls token\nx[:, 0] → [B, 768]"]
        I --> J["LN Post → [B, 768]"]
        J --> K["Projection @ [768,512]\n→ [B, 512]"]
    end

    subgraph Text Tower [Text Transformer]
        P["Input Tokens\n[B, 77]"] --> Q["Token Embedding\n→ [B, 77, 512]"]
        Q --> R["Add Pos Emb\n→ [B, 77, 512]"]
        R --> S["Permute → [77, B, 512]"]
        S --> T["Transformer ×12\n→ [77, B, 512]"]
        T --> U["Permute back → [B, 77, 512]"]
        U --> V["LN Final → [B, 77, 512]"]
        V --> W["Gather EOS token\n(text.argmax) → [B, 512]"]
        W --> X["Projection @ [512,512]\n→ [B, 512]"]
    end

    K --> Y["L2 Normalize\n→ [B, 512]"]
    X --> Z["L2 Normalize\n→ [B, 512]"]
    Y --> AA["Similarity:\nscale · I @ Tᵀ\n→ [B, B]"]
    Z --> AA
    AA --> BB["logits_per_image &\nlogits_per_text"]
```

## :house:2.CLIP_Latest Architecture:house:

The construction process of [MultimodalTransformer](https://github.com/Rtwotwo/Code-Exam/blob/main/dl_exam/vlm/clip_latest/transformer.py) takes the standard Transformer encoder as its core framework, and realizes cross-modal interaction through multi-layer stacked ResidualAttentionBlocks. Each ResidualAttentionBlock contains a custom Attention module (supporting cross-attention mechanism for fusing image and text features), a feed-forward network MLP driven by the [QuickGELU](https://github.com/Rtwotwo/Code-Exam/blob/main/dl_exam/base/utils/activation.py) activation function, and an optional high-precision normalization layer LayerNormFp32 to stabilize the training process. The entire Transformer body is encapsulated in the MultimodalTransformer class, which is responsible for receiving embedding sequences from the visual encoder and text encoder. Under the iterative action of multi-layer residual attention blocks, it gradually aligns and fuses the semantic information of the two modalities, and finally outputs a joint multimodal representation, thereby realizing the modeling of deep semantic association between images and texts. And thanks the [open_clip's](https://github.com/mlfoundations/open_clip.git) open source code.

```mermaid
classDiagram
    direction TB
    
    %% 基础组件层
    class LayerNormFp32 {
        +forward(x:torch.Tensor) torch.Tensor
    }
    class LayerNorm {
        +forward(x:torch.Tensor) torch.Tensor
    }
    class QuickGELU {
        +forward(x:torch.Tensor) torch.Tensor
    }
    class LayerScale {
        +__init__(dim:int, init_values:float, inplace:bool)
        +forward(x:torch.Tensor) torch.Tensor
    }
    class PatchDropout {
        +__init__(prob:float, exclude_first_token:bool)
        +forward(x:torch.Tensor) torch.Tensor
    }
    
    %% 注意力核心层
    class Attention {
        +__init__(dim:int, num_heads:int, qkv_bias:bool, ...)
        +forward(x:torch.Tensor, attn_mask:Optional[torch.Tensor]) torch.Tensor
    }
    class AttentionalPooler {
        +__init__(d_model:int, context_dim:int, n_head:int, ...)
        +forward(x:torch.Tensor) torch.Tensor
    }
    class ResidualAttentionBlock {
        +__init__(d_model:int, n_head:int, mlp_ratio:float, ...)
        +attention(q_x:torch.Tensor, k_x:Optional[torch.Tensor], ...) torch.Tensor
        +forward(q_x:torch.Tensor, k_x:Optional[torch.Tensor], ...) torch.Tensor
    }
    class CustomResidualAttentionBlock {
        +__init__(d_model:int, n_head:int, mlp_ratio:float, ...)
        +forward(x:torch.Tensor, attn_mask:Optional[torch.Tensor]) torch.Tensor
    }
    
    %% Transformer 封装层
    class CustomTransformer {
        +__init__(width:int, layers:int, heads:int, ...)
        +forward(x:torch.Tensor, attn_mask:Optional[torch.Tensor]) torch.Tensor
    }
    class Transformer {
        +__init__(width:int, layers:int, heads:int, ...)
        +forward(x:torch.Tensor, attn_mask:Optional[torch.Tensor]) torch.Tensor
    }
    
    %% 多模态核心层（VisionTransformer 是 CLIP/多模态的视觉塔）
    class VisionTransformer {
        +__init__(image_size:int, patch_size:int, width:int, ...)
        +forward(x:torch.Tensor) torch.Tensor
        +_embeds(x:torch.Tensor) torch.Tensor
        +_pool(x:torch.Tensor) Tuple[torch.Tensor, torch.Tensor]
    }

    %% 继承/依赖关系
    LayerNormFp32 --|> nn.LayerNorm
    LayerNorm --|> nn.LayerNorm
    QuickGELU --|> nn.Module
    LayerScale --|> nn.Module
    PatchDropout --|> nn.Module
    Attention --|> nn.Module
    AttentionalPooler --|> nn.Module
    ResidualAttentionBlock --|> nn.Module
    CustomResidualAttentionBlock --|> nn.Module
    CustomTransformer --|> nn.Module
    Transformer --|> nn.Module
    VisionTransformer --|> nn.Module

    %% 组合依赖（A 使用 B）
    ResidualAttentionBlock --> LayerNorm : uses
    ResidualAttentionBlock --> LayerScale : uses
    ResidualAttentionBlock --> nn.MultiheadAttention : uses
    ResidualAttentionBlock --> QuickGELU : uses (act_layer)
    
    CustomResidualAttentionBlock --> LayerNorm : uses
    CustomResidualAttentionBlock --> LayerScale : uses
    CustomResidualAttentionBlock --> Attention : uses
    CustomResidualAttentionBlock --> QuickGELU : uses (act_layer)
    
    CustomTransformer --> CustomResidualAttentionBlock : contains (resblocks)
    Transformer --> ResidualAttentionBlock : contains (resblocks)
    Transformer --> CustomResidualAttentionBlock : contains (resblocks, if block_type=custom)
    
    VisionTransformer --> PatchDropout : uses
    VisionTransformer --> LayerNorm : uses (ln_pre/ln_post)
    VisionTransformer --> Transformer : contains
    VisionTransformer --> AttentionalPooler : uses (if attentional_pool=True)
    VisionTransformer --> LayerScale : uses (via Transformer)
```

```mermaid
flowchart LR
    subgraph 图像塔 [Vision Tower]
        A["输入图像: [B, 3, H, W]"] --> B["Patch Embedding (Conv2d): [B, D, G, G]"]
        B --> C["Reshape + Permute: [B, G*G, D]"]
        C --> D["拼接 Class Token: [B, G*G+1, D]"]
        D --> E["加位置编码: [B, G*G+1, D]"]
        E --> F["PatchDropout (训练时): [B, N≤G*G+1, D]"]
        F --> G["LN Pre: [B, N, D]"]
        G --> H["Transformer Blocks × L:\n每层保持 [B, N, D]"]
        H --> I{"池化策略?"}
        I -->|tok| J["取 cls token: [B, D]"]
        I -->|avg| K["平均池化 patch: [B, D]"]
        I -->|attn_pool| L["AttentionalPooler:\n[B, Q, D_out] → [B, D_out]"]
        J --> M["LN Post: [B, D]"]
        K --> M
        L --> M
        M --> N["投影 Proj: [B, D] @ [D, D_out] → [B, D_out]"]
        N --> O["图像嵌入: [B, D_out]"]
    end

    subgraph 文本塔 [Text Tower]
        P["输入文本 IDs: [B, L]"] --> Q["Token Embedding: [B, L, D]"]
        Q --> R["加位置编码: [B, L, D]"]
        R --> S["Transformer Blocks × L:\n每层保持 [B, L, D]"]
        S --> T["取 [EOS] token: [B, D]"]
        T --> U["LN Final: [B, D]"]
        U --> V["投影 Proj: [B, D] @ [D, D_out] → [B, D_out]"]
        V --> W["文本嵌入: [B, D_out]"]
    end

    O --> X["计算相似度: [B, B]"]
    W --> X
```

## :house: 3.CoCa Architecture :house:

CoCa ([Contrastive Captioners](https://github.com/Rtwotwo/Code-Exam/blob/main/dl_exam/vlm/clip_latest/coca_model.py)) is a multimodal model that integrates contrastive learning with generative image captioning. Its construction is centered on a dual-tower plus decoder architecture of "vision tower + text encoder + text decoder": first, it parses three types of configurations, namely CLIPTextCfg, CLIPVisionCfg, and MultimodalCfg, and respectively constructs the VisionTransformer visual encoder through _build_vision_tower, the CLIP-style text encoder through build_text_tower, and the MultimodalTransformer text decoder through build_text_decoder_tower. At the same time, it initializes parameters such as the temperature coefficient logit_scale for contrastive learning. The core function of this model is to achieve image-text feature alignment through cross-modal contrastive learning, supporting image-text retrieval/matching tasks. Additionally, relying on visual features to drive the text decoder, it completes generative tasks such as image captioning. It possesses both feature discriminability and generativeness, and is suitable for a wide range of downstream multimodal tasks such as visual question answering and multimodal retrieval.

```mermaid
classDiagram
    direction TB
    %% 基础工具/函数（简化为类关联）
    class AllGather {
        +forward(ctx, x)
        +backward(ctx, grads)
    }
    class EmbedToLatents {
        +__init__(dim, dim_latents)
        +forward(x)
    }
    class LayerNorm {
        +__init__(dim)
        +forward(x)
    }
    class Residual {
        +__init__(fn)
        +forward(x, *args, **kwargs)
    }
    class RotaryEmbedding {
        +__init__(dim)
        +forward(max_seq_len, *, device)
    }
    class SwiGLU {
        +forward(x)
    }

    %% 核心注意力/Transformer 组件
    class ParallelTransformerBlock {
        +__init__(dim, dim_head=64, heads=8, ff_mult=4)
        +get_mask(n, device)
        +get_rotary_embedding(n, device)
        +forward(x, attn_mask=None)
    }
    class CrossAttention {
        +__init__(dim, context_dim=None, dim_head=64, heads=8, parallel_ff=False, ff_mult=4, norm_context=False)
        +forward(x, context)
    }

    %% 核心模型类
    class CoCa {
        +__init__(dim, num_tokens, unimodal_depth, multimodal_depth, dim_latents=None, image_dim=None, num_img_queries=256, dim_head=64, heads=8, ff_mult=4, img_encoder=None, caption_loss_weight=1., contrastive_loss_weight=1., pad_id=0)
        +embed_text(text)
        +embed_image(images=None, image_tokens=None)
        +forward(text, images=None, image_tokens=None, labels=None, return_loss=False, return_embeddings=False)
    }

    %% 继承关系
    LayerNorm --|> nn.Module
    Residual --|> nn.Module
    EmbedToLatents --|> nn.Module
    RotaryEmbedding --|> nn.Module
    SwiGLU --|> nn.Module
    ParallelTransformerBlock --|> nn.Module
    CrossAttention --|> nn.Module
    CoCa --|> nn.Module
    AllGather --|> Function

    %% 组合依赖关系
    CoCa --> LayerNorm : uses (多处归一化)
    CoCa --> Residual : uses (残差连接)
    CoCa --> EmbedToLatents : contains (img_to_latents/text_to_latents)
    CoCa --> RotaryEmbedding : uses (via ParallelTransformerBlock)
    CoCa --> SwiGLU : uses (via ParallelTransformerBlock/CrossAttention)
    CoCa --> ParallelTransformerBlock : contains (unimodal_layers/multimodal_layers)
    CoCa --> CrossAttention : contains (img_attn_pool/multimodal_layers)
    CoCa --> AllGather : uses (分布式场景下特征聚合)
    
    ParallelTransformerBlock --> LayerNorm : uses (归一化)
    ParallelTransformerBlock --> RotaryEmbedding : contains (rotary_emb)
    ParallelTransformerBlock --> SwiGLU : uses (前馈网络)
    
    CrossAttention --> LayerNorm : uses (查询/上下文归一化)
    CrossAttention --> SwiGLU : uses (并行前馈网络)
    
    EmbedToLatents --> nn.Linear : uses (特征投影)
```
