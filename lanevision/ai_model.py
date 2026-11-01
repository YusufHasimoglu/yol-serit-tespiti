"""Kompakt U-Net benzeri şerit piksel segmentasyon modeli.

PyTorch isteğe bağlıdır: klasik ve renk tabanlı modlar torch olmadan çalışır.
"""
from pathlib import Path
import cv2
import numpy as np

INPUT_WIDTH, INPUT_HEIGHT = 256, 144
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / 'modeller' / 'serit_unet_demo.pt'
LEGACY_LAYOUT = ('left', 'right')
STYLE_LAYOUT = ('left_solid', 'left_dashed', 'right_solid', 'right_dashed')


def build_model(output_channels: int = 2):
    """Create the U-Net with a checkpoint-compatible output head.

    Older demo weights contain two left/right channels.  The real-data model
    adds four channels so it can distinguish solid from dashed paint on each
    ego-lane boundary without making old local checkpoints unusable.
    """
    if output_channels not in {2, 4}:
        raise ValueError('U-Net çıkış kanalı 2 veya 4 olmalıdır.')
    import torch
    from torch import nn
    class Conv(nn.Module):
        def __init__(self, inc, out):
            super().__init__()
            self.seq = nn.Sequential(nn.Conv2d(inc, out, 3, padding=1),nn.BatchNorm2d(out),nn.ReLU(inplace=True),
                                     nn.Conv2d(out, out, 3, padding=1),nn.BatchNorm2d(out),nn.ReLU(inplace=True))
        def forward(self,x):return self.seq(x)
    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=Conv(3,12);self.b=Conv(12,24);self.c=Conv(24,48);self.d=Conv(48,64)
            self.u2=Conv(64+48,48);self.u1=Conv(48+24,24);self.u0=Conv(24+12,12)
            self.head=nn.Conv2d(12,output_channels,1);self.pool=nn.MaxPool2d(2)
        def forward(self,x):
            import torch.nn.functional as F
            a=self.a(x);b=self.b(self.pool(a));c=self.c(self.pool(b));d=self.d(self.pool(c))
            u=self.u2(__import__('torch').cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
            u=self.u1(__import__('torch').cat([F.interpolate(u,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
            u=self.u0(__import__('torch').cat([F.interpolate(u,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
            return self.head(u)
    return SmallUNet()


class AISegmenter:
    def __init__(self, weights=DEFAULT_WEIGHTS, device='cpu', threshold=0.42):
        import torch
        weights=Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f'Eğitilmiş model bulunamadı: {weights}. Önce egit.py çalıştırın.')
        self.device=torch.device(device)
        checkpoint=torch.load(str(weights),map_location=self.device,weights_only=True)
        state=checkpoint['model'] if isinstance(checkpoint,dict) and 'model' in checkpoint else checkpoint
        # A saved metadata field is preferred, but shape inspection keeps
        # hand-copied and pre-metadata checkpoints working as well.
        saved_channels=(checkpoint.get('output_channels') if isinstance(checkpoint,dict) else None)
        if saved_channels is None:
            saved_channels=int(state['head.weight'].shape[0])
        self.output_channels=int(saved_channels)
        saved_layout=(checkpoint.get('channel_layout') if isinstance(checkpoint,dict) else None)
        self.channel_layout=tuple(saved_layout or (STYLE_LAYOUT if self.output_channels==4 else LEGACY_LAYOUT))
        if len(self.channel_layout)!=self.output_channels:
            raise ValueError('Model kanal düzeni ağırlıklarla uyuşmuyor.')
        self.model=build_model(self.output_channels).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.threshold=float(threshold)
        self.meta={k:v for k,v in checkpoint.items() if k!='model'} if isinstance(checkpoint,dict) else {}
        self.meta.setdefault('output_channels',self.output_channels)
        self.meta.setdefault('channel_layout',list(self.channel_layout))

    def probabilities(self, bgr):
        import torch
        h,w=bgr.shape[:2]
        rgb=cv2.cvtColor(cv2.resize(bgr,(INPUT_WIDTH,INPUT_HEIGHT)),cv2.COLOR_BGR2RGB)
        t=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().div_(255).to(self.device)
        with torch.inference_mode():
            arr=self.model(t).sigmoid()[0].cpu().numpy().transpose(1,2,0)
        return cv2.resize(arr,(w,h),interpolation=cv2.INTER_LINEAR)

    def masks(self,bgr):
        arr=self.probabilities(bgr)
        return [(arr[:,:,i]>self.threshold).astype(np.uint8)*255 for i in range(self.output_channels)],arr
