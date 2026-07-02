import os
"""
model_fusion_multi.py — Multi-output AQNet-Fusion
==================================================
원본 single-output Fusion 모델을 3/6종 동시 예측으로 확장

구조:
  S2 + S5P → satellite features
  Tabular → LGBM leaf embedding + MLP → tabular features  
  Cross-attention + Concat 하이브리드 융합
  → prediction_count개 오염물질 동시 예측

각 오염물질별 LGBM 예측값을 final output에 직접 더함 (gamma)
"""

import torch
from torchvision.models import mobilenet_v3_small
import torch.nn as nn
import numpy as np
import joblib


class TabularDTEncoder(nn.Module):
    """LGBM leaf embedding + prediction encoder"""
    def __init__(self, dt_model_path, embedding_dim=128, mlp_hidden_dim=128,
                 mlp_layers=3, n_leaf_emb=96):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lgb_models = joblib.load(dt_model_path)
        assert isinstance(self.lgb_models, list)

        self.n_trees = min([m.n_estimators_ for m in self.lgb_models])
        self.n_leaf_emb = n_leaf_emb
        self.leaf_emb_layers = nn.ModuleList([
            nn.Embedding(100, n_leaf_emb) for _ in range(self.n_trees)
        ])
        self.leaf_post = nn.Sequential(
            nn.Linear(self.n_trees * n_leaf_emb, embedding_dim),
            nn.LayerNorm(embedding_dim), nn.ReLU(), nn.Dropout(0.4)
        )
        self.linear = nn.Linear(1, embedding_dim)
        mlp_modules = []
        input_dim = embedding_dim
        for _ in range(mlp_layers - 1):
            mlp_modules += [
                nn.Linear(input_dim, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim), nn.ReLU(), nn.Dropout(0.4)
            ]
            input_dim = mlp_hidden_dim
        mlp_modules.append(nn.Linear(input_dim, embedding_dim))
        self.mlp = nn.Sequential(*mlp_modules)
        self.tab_fusion_weight = nn.Parameter(torch.zeros(2))

    def forward(self, x_tab, return_pred=True):
        device = next(self.parameters()).device
        if isinstance(x_tab, torch.Tensor):
            x_tab_np = x_tab.cpu().numpy()
        else:
            x_tab_np = x_tab

        preds = np.stack([m.predict(x_tab_np) for m in self.lgb_models], axis=0)
        mean_pred = np.mean(preds, axis=0)
        pred_emb = self.linear(torch.FloatTensor(mean_pred).unsqueeze(1).to(device))
        pred_emb = self.mlp(pred_emb)

        leaf_idx = self.lgb_models[0].predict(x_tab_np, pred_leaf=True)[:, :self.n_trees]
        embeds = []
        for t in range(self.n_trees):
            e = self.leaf_emb_layers[t](torch.LongTensor(leaf_idx[:, t]).to(device))
            embeds.append(e)
        embeds = torch.stack(embeds, dim=1).view(leaf_idx.shape[0], -1)
        leaf_emb = self.leaf_post(embeds)

        w = torch.softmax(self.tab_fusion_weight, dim=0)
        tab_feat = w[0] * leaf_emb + w[1] * pred_emb

        if return_pred:
            mean_pred_t = torch.from_numpy(mean_pred.astype(np.float32)).to(device).unsqueeze(1)
            return tab_feat, mean_pred_t
        return tab_feat


class TabularMLPEncoder(nn.Module):
    def __init__(self, input_dim=20, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, out_dim)
        self.ln3 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.4)
        self.out_reg = nn.Linear(out_dim, 1)
        self.proj_input = nn.Linear(input_dim, out_dim) if input_dim != out_dim else nn.Identity()

    def forward(self, x_tab):
        h = torch.relu(self.ln1(self.fc1(x_tab)))
        h = self.dropout(h)
        h = torch.relu(self.ln2(self.fc2(h)))
        h = self.dropout(h)
        h = torch.relu(self.ln3(self.fc3(h)))
        h = h + self.proj_input(x_tab)
        feat = self.dropout(h)
        pred = self.out_reg(feat)
        return feat, pred


class MultiOutputFusion(nn.Module):
    """Multi-pollutant Fusion model with per-pollutant LGBM predictions"""
    def __init__(self, backbone_S2, backbone_S5P, head_sat,
                 backbone_tabular_lgbm, backbone_tabular_mlp,
                 lgbm_encoders_per_pollutant,  # dict: {pollutant: TabularDTEncoder}
                 head_features, tabular_features, prediction_count, pollutant_names):
        super().__init__()
        self.backbone_S2 = backbone_S2
        self.backbone_S5P = backbone_S5P
        self.head_sat = head_sat
        self.backbone_tabular_lgbm = backbone_tabular_lgbm  # primary (NO2)
        self.backbone_tabular_mlp = backbone_tabular_mlp
        self.prediction_count = prediction_count
        self.pollutant_names = pollutant_names

        # Per-pollutant LGBM encoders (for direct prediction addition)
        self.lgbm_encoders = nn.ModuleDict()
        for pol, enc in lgbm_encoders_per_pollutant.items():
            self.lgbm_encoders[pol] = enc

        # Cross-attention
        self.proj_q = nn.Linear(head_features, head_features)
        self.proj_kv = nn.Linear(head_features, head_features)
        self.proj_tab = nn.Linear(tabular_features, head_features)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=head_features, num_heads=12, batch_first=True, dropout=0.45)
        self.ln1 = nn.LayerNorm(head_features)
        self.ffn = nn.Sequential(
            nn.Linear(head_features, head_features), nn.ReLU(),
            nn.Linear(head_features, head_features))
        self.ln2 = nn.LayerNorm(head_features)

        # Concat path
        self.concat_mlp = nn.Sequential(
            nn.Linear(head_features + tabular_features, 256), nn.ReLU(),
            nn.LayerNorm(256), nn.Dropout(0.45),
            nn.Linear(256, head_features), nn.ReLU(), nn.LayerNorm(head_features))

        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(-0.5))
        self.beta = nn.Parameter(torch.tensor(4.0))
        self.gamma = nn.Parameter(torch.tensor(0.4))
        self.tabular_fusion_weight = nn.Parameter(torch.tensor([0.0, 0.0]))

        # Multi-output prediction head
        self.pred_mlp = nn.Sequential(
            nn.Linear(head_features, 128), nn.ReLU(), nn.LayerNorm(128), nn.Dropout(0.45),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64), nn.Dropout(0.45),
            nn.Linear(64, prediction_count))

    def forward(self, x):
        # Satellite features
        f_s2 = self.backbone_S2(x['img'])
        f_s5p = self.backbone_S5P(x['s5p'])
        sat_feat = self.head_sat(torch.cat((f_s2, f_s5p), dim=1))

        # Tabular fusion (primary LGBM + MLP)
        tab_feat_lgbm, _ = self.backbone_tabular_lgbm(x['tabular'], return_pred=True)
        tab_feat_mlp, _ = self.backbone_tabular_mlp(x['tabular'])
        w = torch.softmax(self.tabular_fusion_weight, dim=0)
        tab_feat = w[0] * tab_feat_lgbm + w[1] * tab_feat_mlp

        # Cross-attention path
        q = self.proj_q(sat_feat).unsqueeze(1)
        sat_k = self.proj_kv(sat_feat).unsqueeze(1)
        tab_k = self.proj_tab(tab_feat).unsqueeze(1)
        kv = torch.cat((sat_k, tab_k), dim=1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        h1 = self.ln1(sat_feat + attn_out.squeeze(1))
        attn_path = self.ln2(h1 + self.ffn(h1))

        # Concat path
        beta = torch.nn.functional.softplus(self.beta)
        concat_in = torch.cat((sat_feat, tab_feat * beta), dim=1)
        concat_path = self.concat_mlp(concat_in) + sat_feat

        # Hybrid mix
        alpha = torch.sigmoid(self.alpha)
        fused = alpha * attn_path + (1 - alpha) * concat_path

        # Main prediction
        pred = self.pred_mlp(fused)  # (batch, prediction_count)

        # Per-pollutant LGBM prediction addition
        gamma = torch.sigmoid(self.gamma)
        device = pred.device
        for i, pol in enumerate(self.pollutant_names):
            if pol in self.lgbm_encoders:
                _, lgbm_pred = self.lgbm_encoders[pol](x['tabular'], return_pred=True)
                pred[:, i:i+1] = pred[:, i:i+1] + gamma * lgbm_pred

        return tuple(pred[:, i] for i in range(self.prediction_count))


def get_model(device, network, tabular_switch, S5p_switch, checkpoint=None,
              prediction_count=3, tabular_input_count=20,
              lgbm_dir=None, pollutant_names=None, n_runs=1, run=1):
    """
    Args:
        lgbm_dir: LGBM 모델 폴더 (각 오염물질별 하위 폴더)
        pollutant_names: ["no2", "o3", "pm10"] 등
        run: 현재 run 번호
    """
    S2_num_features = 320
    S5p_num_features = 128
    tabular_features = 128
    head_features = 48

    if pollutant_names is None:
        pollutant_names = ["no2", "o3", "pm10"]

    # S2 backbone
    backbone_S2 = mobilenet_v3_small(pretrained=checkpoint, num_classes=1000)
    backbone_S2.features[0][0] = nn.Conv2d(12, 16, 3, 1, 1)
    backbone_S2.classifier[3] = nn.Linear(1024, S2_num_features)

    # S5P backbone
    backbone_S5P = nn.Sequential(
        nn.Conv2d(1, 10, 3), nn.ReLU(), nn.MaxPool2d(3),
        nn.Conv2d(10, 15, 5), nn.ReLU(), nn.MaxPool2d(3),
        nn.Flatten(), nn.Linear(1815, S5p_num_features))

    # Satellite projection head
    head_sat = nn.Sequential(
        nn.Linear(S2_num_features + S5p_num_features, 384), nn.ReLU(),
        nn.Linear(384, 192), nn.ReLU(),
        nn.Linear(192, head_features))

    if tabular_switch and lgbm_dir is not None:
        # Primary LGBM encoder (첫 번째 오염물질, 보통 NO2)
        primary_pol = pollutant_names[0]
        primary_path = os.path.join(lgbm_dir, primary_pol, f"lgbm_models_run{run}.pkl")
        backbone_tabular_lgbm = TabularDTEncoder(
            dt_model_path=primary_path,
            embedding_dim=tabular_features, mlp_hidden_dim=128, mlp_layers=3)

        backbone_tabular_mlp = TabularMLPEncoder(
            input_dim=tabular_input_count, out_dim=tabular_features)

        # Per-pollutant LGBM encoders
        lgbm_encoders = {}
        for pol in pollutant_names:
            pol_path = os.path.join(lgbm_dir, pol, f"lgbm_models_run{run}.pkl")
            if os.path.exists(pol_path):
                lgbm_encoders[pol] = TabularDTEncoder(
                    dt_model_path=pol_path,
                    embedding_dim=tabular_features, mlp_hidden_dim=128, mlp_layers=3)

        model = MultiOutputFusion(
            backbone_S2, backbone_S5P, head_sat,
            backbone_tabular_lgbm, backbone_tabular_mlp,
            lgbm_encoders,
            head_features, tabular_features,
            prediction_count, pollutant_names)
    else:
        # Tabular 없이 satellite only
        head_2 = nn.Sequential(
            nn.Linear(S2_num_features + S5p_num_features, 384), nn.ReLU(),
            nn.Linear(384, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.Dropout(0.35), nn.ReLU(),
            nn.Linear(64, 16), nn.Dropout(0.35), nn.ReLU(),
            nn.Linear(16, prediction_count))

        model = _SatelliteOnlyModel(backbone_S2, backbone_S5P, head_2, prediction_count)

    return model


class _SatelliteOnlyModel(nn.Module):
    """Satellite-only fallback (no tabular)"""
    def __init__(self, backbone_S2, backbone_S5P, head, prediction_count):
        super().__init__()
        self.backbone_S2 = backbone_S2
        self.backbone_S5P = backbone_S5P
        self.head = head
        self.prediction_count = prediction_count

    def forward(self, x):
        f_s2 = self.backbone_S2(x['img'])
        f_s5p = self.backbone_S5P(x['s5p'])
        pred = self.head(torch.cat((f_s2, f_s5p), dim=1))
        return tuple(pred[:, i] for i in range(self.prediction_count))
