"""형태 정리(cleanup) + 단면 채색 유틸.

시뮬레이터가 확률장을 라벨로 바꾼 뒤 쓰는 보조 도구다. 정리는 '미세 잡음
제거'와 '목표 비율 임계화'에 한정하며, 확률장에 담긴 마블링 구조 자체는
보존한다(새 구조를 만들지 않는다).
"""

import numpy as np
from scipy import ndimage


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def threshold_to_ratio(p_fat, mask, target_ratio):
    """확률장 상위 ``target_ratio`` 분위수로 임계화 → 지방 boolean 마스크.

    확률이 높은 voxel 부터 지방으로 채워 목표 면적비를 정확히 맞춘다.
    반환: (fat_mask, threshold).
    """
    mask = mask.astype(bool)
    vals = p_fat[mask]
    if vals.size == 0:
        return np.zeros_like(mask), 1.0
    target_ratio = float(np.clip(target_ratio, 0.005, 0.95))
    thr = float(np.quantile(vals, 1.0 - target_ratio))
    return (p_fat >= thr) & mask, thr


def remove_small_components(mask, min_size=6):
    """3D 연결요소 중 너무 작은 잡음 덩어리를 제거(구조는 보존)."""
    mask = mask.astype(bool)
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(mask, dtype=np.int32), labels,
                        index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= int(min_size)
    return keep[labels]


def colorize_slice(fat_prob, muscle_shade, fat_cut=0.5):
    """단면을 선홍색 살코기 + 상아색 지방으로 채색(정적 PNG 미리보기용)."""
    h, w = fat_prob.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    fiber = 0.5 + 0.5 * np.sin(xx * 0.16 + yy * 0.025 + muscle_shade * 3.0)
    shade = 0.82 + 0.28 * (muscle_shade - 0.5) + 0.075 * (fiber - 0.5)
    muscle = np.stack([0.66 * shade, 0.16 * shade, 0.15 * shade], axis=-1)

    fat = np.array([0.95, 0.93, 0.89])
    t = smoothstep(fat_cut - 0.075, fat_cut + 0.075, fat_prob)[..., None]
    img = muscle * (1.0 - t) + fat * t
    edge = np.exp(-((fat_prob - fat_cut) ** 2) / (2 * 0.035 ** 2))
    img += edge[..., None] * np.array([0.035, 0.018, 0.010])
    return np.clip(img, 0.0, 1.0)
