import torch, time
def vram(): return f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB"
t=time.time()
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
det = AutoModelForZeroShotObjectDetection.from_pretrained("/root/ckpts/llmdet_large", torch_dtype=torch.float16).cuda()
print("LLMDet OK", vram())
from sam2.build_sam import build_sam2_video_predictor
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", "/root/ckpts/sam2/sam2.1_hiera_large.pt")
print("SAM2 OK", vram())
from moge.model.v2 import MoGeModel
moge = MoGeModel.from_pretrained("/root/ckpts/moge2").cuda()
print("MoGe-2-normal OK", vram(), f"{time.time()-t:.0f}s total")
import open3d; print("Open3D", open3d.__version__)
