"""
transforms_busan.py — 부울경 데이터용 전처리 변환
================================================
원본 transforms_3poll.py 대비 변경점:
  1. DatasetStatistics → 데이터에서 자동 계산 (유럽 하드코딩 제거)
  2. 21개 tabular 피처 정규화 지원
  3. 6종 오염물질 (NO2, O3, PM10, SO2, CO, PM25) 지원
"""

import os
import random
import numpy as np
import torch
from rasterio.plot import reshape_as_image

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"


class ChangeBandOrder(object):
    """BigEarthNet 밴드 순서로 재정렬 + 200×200 → 120×120 center crop"""
    def __call__(self, sample):
        img = sample["img"].copy()
        img = np.moveaxis(img, -1, 0)  # (H,W,12) → (12,H,W)
        reordered_img = np.zeros(img.shape)
        # 입력 순서: B04,B03,B02,B08,B05,B06,B07,B8A,B11,B12,B01,B09
        # 출력 순서: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
        reordered_img[0]  = img[10]  # B01
        reordered_img[1]  = img[2]   # B02
        reordered_img[2]  = img[1]   # B03
        reordered_img[3]  = img[0]   # B04
        reordered_img[4]  = img[4]   # B05
        reordered_img[5]  = img[5]   # B06
        reordered_img[6]  = img[6]   # B07
        reordered_img[7]  = img[3]   # B08
        reordered_img[8]  = img[7]   # B8A
        reordered_img[9]  = img[11]  # B09
        reordered_img[10] = img[8]   # B11
        reordered_img[11] = img[9]   # B12

        if img.shape[1] != 120 or img.shape[2] != 120:
            reordered_img = reordered_img[:, 40:160, 40:160]

        out = {}
        for k, v in sample.items():
            out[k] = reordered_img if k == "img" else v
        return out


class DatasetStatistics(object):
    """
    데이터에서 자동으로 통계를 계산하거나, 제공된 값을 사용
    
    사용법:
        # 방법 1: samples에서 자동 계산
        stats = DatasetStatistics.from_samples(samples)
        
        # 방법 2: 기본값 사용 (나중에 업데이트)
        stats = DatasetStatistics()
    """
    def __init__(self):
        # S2 채널 통계 (BigEarthNet 유럽 기준 — S2 이미지는 동일 위성이므로 유지)
        self.channel_means = np.array([
            340.76769064, 429.9430203, 614.21682446,
            590.23569706, 950.68368468, 1792.46290469,
            2075.46795189, 2218.94553375, 2266.46036911,
            2246.0605464, 1594.42694882, 1009.32729131
        ])
        self.channel_std = np.array([
            554.81258967, 572.41639287, 582.87945694,
            675.88746967, 729.89827633, 1096.01480586,
            1273.45393088, 1365.45589904, 1356.13789355,
            1302.3292881, 1079.19066363, 818.86747235
        ])

        # 나머지는 placeholder (from_samples로 덮어씀)
        self.s5p_mean = 0.0
        self.s5p_std = 1.0

        self.alt_mean = 0.0
        self.alt_std = 1.0

        self.popdense_mean = 0.0
        self.popdense_std = 1.0

        self.lat_mean = 0.0
        self.lat_std = 1.0

        self.lon_mean = 0.0
        self.lon_std = 1.0

        self.temp3yr_mean = 0.0
        self.temp3yr_std = 1.0

        self.wind3yr_mean = 0.0
        self.wind3yr_std = 1.0

        self.precip3yr_mean = 0.0
        self.precip3yr_std = 1.0

        self.rh3yr_mean = 0.0
        self.rh3yr_std = 1.0

        # OSM features
        for feat in ["num_roads_500m", "num_roads_1km",
                     "num_buildings_500m", "num_buildings_1km",
                     "num_factory_500m", "num_factory_1km",
                     "num_industrial_landuse_500m", "num_industrial_landuse_1km"]:
            setattr(self, f"{feat}_mean", 0.0)
            setattr(self, f"{feat}_std", 1.0)

        # 오염물질 6종
        self.no2_mean = 0.0;  self.no2_std = 1.0
        self.o3_mean = 0.0;   self.o3_std = 1.0
        self.pm10_mean = 0.0; self.pm10_std = 1.0
        self.so2_mean = 0.0;  self.so2_std = 1.0
        self.co_mean = 0.0;   self.co_std = 1.0
        self.pm25_mean = 0.0; self.pm25_std = 1.0

    @classmethod
    def from_samples(cls, samples):
        """samples 리스트에서 통계 자동 계산"""
        stats = cls()

        def _mean_std(key):
            vals = [s[key] for s in samples if key in s and s[key] is not None
                    and not (isinstance(s[key], float) and np.isnan(s[key]))]
            if not vals:
                return 0.0, 1.0
            vals = np.array(vals, dtype=np.float64)
            m, s = vals.mean(), vals.std()
            return float(m), float(max(s, 1e-8))

        # S5P
        s5p_vals = [s["s5p"].mean() for s in samples if "s5p" in s and s["s5p"] is not None]
        if s5p_vals:
            s5p_all = np.concatenate([s["s5p"].flatten() for s in samples
                                      if "s5p" in s and s["s5p"] is not None])
            stats.s5p_mean = float(np.nanmean(s5p_all))
            stats.s5p_std = float(max(np.nanstd(s5p_all), 1e-8))

        # Tabular
        stats.alt_mean, stats.alt_std = _mean_std("Altitude")
        stats.popdense_mean, stats.popdense_std = _mean_std("PopulationDensity")
        stats.lat_mean, stats.lat_std = _mean_std("Latitude")
        stats.lon_mean, stats.lon_std = _mean_std("Longitude")
        stats.temp3yr_mean, stats.temp3yr_std = _mean_std("Temp_3yr")
        stats.wind3yr_mean, stats.wind3yr_std = _mean_std("Wind_3yr")
        stats.precip3yr_mean, stats.precip3yr_std = _mean_std("Precip_3yr")
        stats.rh3yr_mean, stats.rh3yr_std = _mean_std("RH_3yr")

        # OSM + 신규 피처 (동적 처리)
        _dynamic_features = [
            # OSM (8)
            "num_roads_500m", "num_roads_1km",
            "num_buildings_500m", "num_buildings_1km",
            "num_factory_500m", "num_factory_1km",
            "num_industrial_landuse_500m", "num_industrial_landuse_1km",
            # 기상 신규 (7)
            "SolarRad_3yr", "ClearSkyRad_3yr", "Dewpoint_3yr",
            "SkinTemp_3yr", "TempRange_3yr", "SpecHumidity_3yr", "CloudAmt_3yr",
            # 선박 밀도 (3)
            "ship_density_1km", "ship_density_5km", "ship_density_10km",
            # 측정소 유형 (6)
            "rural", "suburban", "urban", "traffic", "industrial", "background",
        ]
        for feat in _dynamic_features:
            m, s = _mean_std(feat)
            setattr(stats, f"{feat}_mean", m)
            setattr(stats, f"{feat}_std", s)

        # 오염물질 6종
        stats.no2_mean, stats.no2_std = _mean_std("no2")
        stats.o3_mean, stats.o3_std = _mean_std("o3")
        stats.pm10_mean, stats.pm10_std = _mean_std("pm10")
        stats.so2_mean, stats.so2_std = _mean_std("so2")
        stats.co_mean, stats.co_std = _mean_std("co")
        stats.pm25_mean, stats.pm25_std = _mean_std("pm25")

        print(f"[DatasetStatistics] 자동 계산 완료:")
        print(f"  S5P: mean={stats.s5p_mean:.4e}, std={stats.s5p_std:.4e}")
        print(f"  NO2: mean={stats.no2_mean:.2f}, std={stats.no2_std:.2f}")
        print(f"  O3:  mean={stats.o3_mean:.2f}, std={stats.o3_std:.2f}")
        print(f"  PM10: mean={stats.pm10_mean:.2f}, std={stats.pm10_std:.2f}")
        print(f"  SO2: mean={stats.so2_mean:.2f}, std={stats.so2_std:.2f}")
        print(f"  CO:  mean={stats.co_mean:.3f}, std={stats.co_std:.3f}")
        print(f"  PM25: mean={stats.pm25_mean:.2f}, std={stats.pm25_std:.2f}")
        print(f"  Alt: mean={stats.alt_mean:.1f}, std={stats.alt_std:.1f}")
        print(f"  Pop: mean={stats.popdense_mean:.1f}, std={stats.popdense_std:.1f}")

        return stats


class Normalize(object):
    """모든 피처 정규화 (z-score)"""
    def __init__(self, statistics):
        self.statistics = statistics

    def _norm(self, val, mean, std):
        return np.array((val - mean) / std)

    def __call__(self, sample):
        s = self.statistics
        out = {}

        for k, v in sample.items():
            if k == "img":
                img = reshape_as_image(v.copy())
                out[k] = np.moveaxis(
                    (img - s.channel_means) / s.channel_std, -1, 0)
            elif k == "s5p":
                s5p = v.copy()
                # 120×120으로 center crop (모델 입력 크기 고정)
                h, w = s5p.shape[:2]
                if h != 120 or w != 120:
                    ch = (h - 120) // 2
                    cw = (w - 120) // 2
                    s5p = s5p[ch:ch+120, cw:cw+120]
                out[k] = np.array((s5p - s.s5p_mean) / s.s5p_std)
            elif k == "no2":
                out[k] = self._norm(v, s.no2_mean, s.no2_std)
            elif k == "o3":
                out[k] = self._norm(v, s.o3_mean, s.o3_std)
            elif k == "pm10":
                out[k] = self._norm(v, s.pm10_mean, s.pm10_std)
            elif k == "so2":
                out[k] = self._norm(v, s.so2_mean, s.so2_std)
            elif k == "co":
                out[k] = self._norm(v, s.co_mean, s.co_std)
            elif k == "pm25":
                out[k] = self._norm(v, s.pm25_mean, s.pm25_std)
            elif k == "Altitude":
                out[k] = self._norm(v, s.alt_mean, s.alt_std)
            elif k == "PopulationDensity":
                out[k] = self._norm(v, s.popdense_mean, s.popdense_std)
            elif k == "Latitude":
                out[k] = self._norm(v, s.lat_mean, s.lat_std)
            elif k == "Longitude":
                out[k] = self._norm(v, s.lon_mean, s.lon_std)
            elif k == "Temp_3yr":
                out[k] = self._norm(v, s.temp3yr_mean, s.temp3yr_std)
            elif k == "Wind_3yr":
                out[k] = self._norm(v, s.wind3yr_mean, s.wind3yr_std)
            elif k == "Precip_3yr":
                out[k] = self._norm(v, s.precip3yr_mean, s.precip3yr_std)
            elif k == "RH_3yr":
                out[k] = self._norm(v, s.rh3yr_mean, s.rh3yr_std)
            elif hasattr(s, f"{k}_mean"):
                m = getattr(s, f"{k}_mean", 0)
                sd = getattr(s, f"{k}_std", 1)
                out[k] = self._norm(v, m, sd)
            else:
                out[k] = v
        return out

    # undo standardization (for evaluation)
    @staticmethod
    def undo_no2_standardization(s, v): return v * s.no2_std + s.no2_mean
    @staticmethod
    def undo_o3_standardization(s, v): return v * s.o3_std + s.o3_mean
    @staticmethod
    def undo_pm10_standardization(s, v): return v * s.pm10_std + s.pm10_mean
    @staticmethod
    def undo_so2_standardization(s, v): return v * s.so2_std + s.so2_mean
    @staticmethod
    def undo_co_standardization(s, v): return v * s.co_std + s.co_mean
    @staticmethod
    def undo_pm25_standardization(s, v): return v * s.pm25_std + s.pm25_mean


class ToTensor(object):
    """numpy → torch tensor 변환"""
    TENSOR_KEYS = {"img", "s5p", "no2", "o3", "pm10", "so2", "co", "pm25"}

    def __call__(self, sample):
        out = {}
        for k, v in sample.items():
            if k in self.TENSOR_KEYS and v is not None:
                out[k] = torch.from_numpy(np.array(v).copy())
            else:
                out[k] = v
        return out


class Randomize(object):
    """S2/S5P 이미지 랜덤 augmentation (flip, rotate)"""
    def __call__(self, sample):
        img = sample.get("img").copy()
        s5p = sample.get("s5p")
        s5p_available = s5p is not None
        if s5p_available:
            s5p = s5p.copy()

        if random.random() > 0.5:
            img = np.flip(img, 1)
            if s5p_available: s5p = np.flip(s5p, 0)
        if random.random() > 0.5:
            img = np.flip(img, 2)
            if s5p_available: s5p = np.flip(s5p, 1)
        if random.random() > 0.5:
            k = np.random.randint(0, 4)
            img = np.rot90(img, k, axes=(1, 2))
            if s5p_available: s5p = np.rot90(s5p, k, axes=(0, 1))

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v
        return out
