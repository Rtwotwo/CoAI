# :rocket:Model Introduction:rocket:

Vision-Language Models (VLMs), as the core carrier of multimodal artificial intelligence, have made significant breakthroughs in recent years in terms of architectural design, training paradigms, and application scenarios. Modern VLMs are based on Transformers as their backbone, establishing a cross-modal semantic foundation through large-scale image-text alignment pre-training (such as CLIP and ALIGN), and have gradually evolved into general multimodal bases that support fine-grained understanding, generation, and reasoning. Current mainstream VLM architectures generally adopt a "two-tower" or "fusion encoder" structure. During the pre-training phase, they combine contrastive learning, mask modeling, and generative objectives to achieve deep alignment between images and text. In the post-alignment phase, they enhance the model's generalization and controllability in open-world tasks through instruction tuning, reinforcement learning from human feedback (RLHF), and domain adaptation strategies. Cutting-edge VLMs not only support traditional tasks such as image-text retrieval, Visual Question Answering (VQA), and image captioning but also demonstrate strong zero-shot transfer capabilities, multi-step reasoning abilities (such as Chain-of-Thought in VLM), as well as the potential for tool invocation and interaction with external knowledge. Representative models include: BLIP/BLIP-2 (efficient modular design), Flamingo (few-shot learning based on frozen pre-trained models), KOSMOS-1/2 (unified multimodal sequence modeling), LLaVA/LLaVA-NeXT (open-source multimodal dialogue agents), Qwen-VL/Qwen2-VL (supporting high-resolution and complex layout understanding), IDEFICS2 (open science-oriented multilingual VLM), as well as closed-source system-level models such as Google’s PaLI-X, Gemini series, and OpenAI’s GPT-4V(ision).

## :house:1.CLIP Architecture:house:

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

```mermaid
flowchart LR
    A["原始文本\n(str 或 List[str])"]
    --> B["清洗文本\n(clean_fn: lower/whitespace/canonicalize)\n→ str"]
    --> C["正则分词\nre.findall(pat, text)\n→ List[str]"]
    --> D["UTF-8 编码 → 字节\n.encode('utf-8')\n→ List[int] (0~255)"]
    --> E["字节 → 可打印 Unicode\nbyte_encoder[b]\n→ str"]
    --> F["BPE 子词分割\nbpe(token)\n→ 'sub1 sub2 ...</w>'"]
    --> G["Token → ID\nencoder[subword]\n→ List[int] (len=N)"]
    --> H["添加特殊标记\n[SOT] + ids + [EOT]\n→ len=N+2"]
    --> I["截断/填充至 77\n(末尾强制为 EOT)\n→ List[int] (len≤77)"]
    --> J["输出 LongTensor\n→ torch.LongTensor\nShape: [batch_size, 77]"]
```
