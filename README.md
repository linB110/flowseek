## OS Enviornment and Hardware
Ubuntu : 20.04
GPU : Nvidia RTX 5060 Ti (sm_120)

## Conda env

1. create env

`conda create -n flowseek python=3.10 -y`

---

2. install torch

`pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

---

3. install xformer

pip install xformers

---

4. install other dependencies

note : remove some already installed dependenciew (like torch)

`pip install -r requirements.txt' 

## Download depth_anything_v2_vits.pth

'wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth'

---

## Download flowseek_T_CT.pth

gdown --fuzzy 'https://drive.google.com/file/d/1COOQFkMulzpBm4zMoWsaRGk7E3YcVr2I/view?usp=share_link'


---

## Modify /path/flowseek/core/depth_anything_v2/dinov2_layers/attention.py

'''
class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)
        
        q_half = q.to(dtype=torch.float16)
        k_half = k.to(dtype=torch.float16)
        v_half = v.to(dtype=torch.float16)
        
        #x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        with autocast(device_type="cuda", dtype=torch.float16):
            x = memory_efficient_attention(q_half, k_half, v_half, attn_bias=attn_bias)
        x = x.to(torch.float32)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
'''

---

## Dataset

dataset : KITTI

where : https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow

put downloaded data to /path/flowseek/data/KITTI

---

## Evaluation

run `python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_CT.pth --dataset kitti`

---
