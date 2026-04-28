import numpy as np

import torch
import torch.nn as nn

from nystrom_attention import NystromAttention

import warnings
warnings.filterwarnings("ignore")


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
    
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x
    
class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_P, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        nn.init.normal_(self.cls_token, std=1e-6)
        
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        
        # pad, 后面加了一截，[B, N+L, 512]
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        
        h = torch.cat([features, features[:, :add_length, :]], 
                      dim=1)  
        
        # add cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        
        # Trans layer 1, PPEG, Tran layer 2
        h = self.layer1(h)  # [B, 1+N+L, 512]
        h = self.pos_layer(h, _H, _W)  # [B, 1+N+L, 512]    
        h = self.layer2(h)  # [B, 1+N+L, 512]
        
        # layer norm
        h = self.norm(h)
        
        return h[:, 0], h[:, 1:H+1]




class PathViT(nn.Module):
    def __init__(self,
                 omic_sizes=[100, 200, 300, 400, 500, 600],
                 n_classes=4,
                 num_cluster=6,
                 seed=1,
                 fusion='concat',
                 model_size="small",
                 decoder_embed_dim=128,
                 dropout=0.25,
                 use_kd=False,
                 kd_dim=1024):
        super(PathViT, self).__init__()

        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.num_cluster = num_cluster
        self.seed = seed
        self.use_kd = use_kd
        self.kd_dim = kd_dim

        # pathology modified by pt_file
        self.size_dict = {"path": {"small": [1024, 256, 256],   # luad-plip，得从1024变为512
                                   "trans": [768, 256, 256]},
                          "omics": {"small": [256, 256],
                                    "trans": [256, 256]}}  # [1024, 1024, 1024, 256], BRCA:[1024, 256]

        
        # ====== Pathology Embedding ======
        wsi_hidden = self.size_dict["path"][model_size] 
        fc = []
        for idx in range(len(wsi_hidden) - 1):
            fc.append(nn.Linear(wsi_hidden[idx], wsi_hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(dropout))
        self.pathology_fc = nn.Sequential(*fc)  # 2 layers
     
        # Pathology Transformer
        self.path_encoder = Transformer_P(feature_dim=wsi_hidden[-1])

        # ====== Survival Layer ======
        self.classifier = nn.Linear(wsi_hidden[-1], self.n_classes)
        
        if self.use_kd:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 256))
            nn.init.normal_(self.mask_token, std=.02)

    def forward(self, **kwargs):
        
        x_path = kwargs['x_path'] # [N,1024]
        
        # ====== WSI, Pathology FC ======
        pathology_features = self.pathology_fc(x_path).unsqueeze(0) # [1, N, 256]

        # Apply mask if provided (for iBOT)
        if 'mask' in kwargs and kwargs['mask'] is not None:
             mask = kwargs['mask'].unsqueeze(0) # [1, N], 1 means masked
             
             # Replace masked tokens with mask_token，然后直接往下算
             mask = mask.unsqueeze(-1).expand_as(pathology_features)
             pathology_features = torch.where(mask, self.mask_token.expand_as(pathology_features), pathology_features)
        
        
        # encoder
        # cls token + patch tokens, [1, 256], [1, N, 256]
        cls_token, patch_token = self.path_encoder(pathology_features) 

        
        # predict
        logits = self.classifier(cls_token)  # [1, n_classes]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        # Knowledge Distillation is used
        if self.use_kd:
                          
            # Flatten patch tokens: [1, N, D] -> [1*N, D] for projection
            B, N, D = patch_token.shape
            patch_token_flat = patch_token.reshape(B*N, D)
             
            #  中间两个cls_token, patch_token没多大用
            return logits, hazards, S, (cls_token, patch_token_flat)

        return logits, hazards, S




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define dimensions
    omic_sizes = [100, 200, 300, 400, 500, 600]
    n_classes = 4
    wsi_embedding_dim = 1024

    # Initialize model
    print("Initializing model...")
    model = PathViT(fusion='concat', 
                      omic_sizes=omic_sizes, 
                      n_classes=n_classes,
                      dropout=0.25,
                      use_kd=True).to(device)
    model.eval()

    # 构造 WSI 输入：形状 [N, 1024]
    N = 12817  # patch 数量
    x_path = torch.randn((N, wsi_embedding_dim), dtype=torch.float32).to(device)

    # 构造 6 个 omics 通路输入：每段一个 1D 向量
    x_omic_list = [torch.randn(s, dtype=torch.float32).to(device) for s in omic_sizes]

    print('\n\nStart Forward {}'.format('-' * 60))
    with torch.no_grad():
        outputs = model(x_path=x_path, 
                        x_omic1=x_omic_list[0], x_omic2=x_omic_list[1], x_omic3=x_omic_list[2], 
                        x_omic4=x_omic_list[3], x_omic5=x_omic_list[4], x_omic6=x_omic_list[5])
        
        logits = outputs[0]
        hazards = outputs[1]
        S = outputs[2]

    print(f"Logits shape:  {logits.shape} ")
    print(f"Hazards shape: {hazards.shape} ")
    print(f"Surv shape:    {S.shape} ")
    
    (cls_token, patch_token)=outputs[3]
    print(f"cls token:    {cls_token.shape} ")
    print(f"patch token:    {patch_token.shape} ")
    
    print("Done!{}\n".format('-' * 60))



if __name__ == '__main__':
    """模块测试
    python -m models.pathViT.network
    """

    main()
  
