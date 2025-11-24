"""
Author: Redal
Date: 2025-11-04
Todo: clip.py to download pretrained model
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import os
import urllib
import warnings
import hashlib
from tqdm import tqdm
from packaging import version
from typing import Union, List, Tuple
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, Normalize
from torchvision.transforms import CenterCrop, ToTensor
from model import build_model
from tokenizer import Tokenizer as _Totokenizer
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("Pytorch的版本需要更新至1.7.1以上")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Totokenizer()
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",}


def _download(url:str, root:str):
    """下载预训练的模型权重"""
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    # 提取sha256编码用于校验后续文件完整性
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f'{download_target}目前存在并且不是正常的文件')
    # SHA256文件校验码对比检验
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, 'rb').read()).hexdigest()==expected_sha256:
            return download_target
        else: warnings.warn(f'{download_target}文件存在,但是SHA256编码不匹配,需删除重新下载')
    # 下载预训练权重文件
    # urllib.request.urlopen(url)：创建一个网络连接对象，用于从指定的url获取数据
    # open(download_target, 'wb')：以二进制写入模式打开本地文件，准备保存下载的内容
    with urllib.request.urlopen(url) as source, open(download_target, 'wb') as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer: break
                output.write(buffer)
                loop.update(len(buffer))
    if hashlib.sha256(open(download_target, 'rb').read()).hexdigest() != expected_sha256:
        raise RuntimeError("预训练模型权重已经被下载,但是SHA256编码不匹配,需要重新下载")
    return download_target
        

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(std=(0.48145466, 0.4578275, 0.40821073), 
                mean=(0.26862954, 0.26130258, 0.27577711))])
def available_models()->List[str]:
    """返回所有可能使用到的模型名称"""
    return list(_MODELS.keys())


def load(name:str, device:Union[str, torch.device]='cuda' if torch.cuda.is_available() else 'cpu',
         jit:bool=False, download_root:str=None):
    """加载模型权重给CLIP模型
    name:CLIP模型的名称,可以在_MODELS中查看全部的模型名称;
    jit:是加载优化后的JIT模型,还是加载更便于修改的非JIT模型""" 
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else: raise RuntimeError(f'模型名称{name}不存在,已知提供的模型包括{available_models()}')
    # 处理加载JIT模型的权重
    with open(model_path, 'rb') as open_file:
        try:
            model = torch.jit.load(open_file, map_location=device if jit else 'cpu').eval()
            state_dict = None
        except RuntimeError:
            if jit: warnings.warn(f'模型{model_path}不是JIT模型,尝试加载为非JIT模型')
            state_dict = torch.load(open_file, map_location='cpu')
    # 处理加载非JIT模型的权重
    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == 'cpu': model.float()
        return model, _transform(model.visual.input_resolution)
    # 修补设备名称
    device_holder = torch.jit.trace(lambda:torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findALLNodes("prim::Constant") if "Device" in repr(n)][-1]
    def _node_get(node:torch._C.Node, key:str):
        """获取在返回类型上具有多态性的节点的属性
        来自https://github.com/pytorch/pytorch/pull"""
        sel = node.kindOf(key)
        return getattr(node, sel)(key)
    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)
    # 在CPU上把数据类型修正/转换为float32
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()
        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype 可以作为 aten::to () 的第二个或第三个参数
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()
    return model, _transform(model.input_resolution.item())


def tokenize(texts:Union[str, List[str]], 
            context_length:int=77, 
            truncate:bool=False
            )->Union[torch.IntTensor, torch.LongTensor]:
    """返回给定输入字符串的标记化表示
    texts:Union[str, List[str]]要进行分词的输入字符串或输入字符串列表
    context_length:int要使用的上下文长度,所有CLIP模型都使用77作为上下文长度
    truncate:bool若文本的编码长度超过上下文长度,是否对文本进行截断
    一个包含生成标记的二维张量，形状 = [输入字符串数量，上下文长度]"""
    if isinstance(texts, str): texts = [texts]
    # 将自然语言文本标记化encoding成数字
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    # 针对 PyTorch不同版本的兼容性处理
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else: result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate: 
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else: raise RuntimeError(f'输入{texts[i]}对于模型长度{context_length}太长')
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result