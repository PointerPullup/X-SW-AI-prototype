"""
steak_shape.py
==============
등심(striploin / 등심) 단면 모양의 슬랩(slab) 마스크 생성.

기존엔 직육면체 전체에 마블링을 채웠지만, 실제 스테이크는 (1) 불규칙한 둥근
단면 윤곽을 가지고 (2) 얇은 판(slab) 형태이며 (3) 가장자리에 피하지방 캡
(흰 테두리)이 있다. 예시 이미지의 '복셀 스테이크' 룩을 위해 이 모양을 만든다.

좌표 규약: shape=(nz, ny, nx)
  - z = 두께(thickness, 얇음)
  - x = 등심의 긴 방향(결 방향)
  - y = 등심의 짧은 방향
"""

import numpy as np
from scipy import ndimage


def loin_slab_mask(shape, seed=0, fill=0.62):
    """등심 단면 모양의 3D boolean 마스크 + 부가 정보를 반환.

    in-plane(x-y) 윤곽을 저주파 sin 합성으로 울퉁불퉁한 타원(blob)으로 만들고,
    z 방향으로 그대로 확장해 슬랩을 만든다.

    반환 dict:
      - mask        : (nz,ny,nx) bool. True = 고기(tissue), False = 공기(air)
      - edge_dist   : in-plane 경계까지의 거리(voxel). 지방 캡 배치에 사용.
      - cap_weight  : [0,1] 경계 부근에서 1 → 피하지방 캡 강도(둘레 일부만).
    """
    nz, ny, nx = shape
    rng = np.random.default_rng(seed)

    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    cy, cx = ny / 2.0, nx / 2.0
    ry, rx = ny * 0.43, nx * 0.46

    # 중심 기준 각도/정규화 반경
    ang = np.arctan2((yy - cy) / ry, (xx - cx) / rx)
    norm_r = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)

    # 윤곽을 울퉁불퉁하게: 여러 주파수 sin 합성 (등심 특유의 비대칭 blob)
    ph = rng.uniform(0, 2 * np.pi, 5)
    boundary = (
        1.0
        + 0.16 * np.sin(ang * 2 + ph[0])
        + 0.10 * np.sin(ang * 3 + ph[1])
        + 0.06 * np.sin(ang * 5 + ph[2])
        + 0.04 * np.sin(ang * 7 + ph[3])
    )
    # 한쪽으로 살짝 늘어진 '꼬리(tail)' 한 덩어리 추가
    boundary += 0.12 * np.exp(-((ang - ph[4]) ** 2) / (2 * 0.5 ** 2))

    inside2d = norm_r <= boundary

    # 목표 채움 비율(fill)에 맞게 전체 스케일 미세 조정
    # (blob 크기를 살짝 키우거나 줄여서 평균 fill 근처로)
    cur = inside2d.mean()
    if cur > 1e-6:
        scale = np.sqrt(fill / cur)
        inside2d = norm_r <= boundary * scale

    mask2d = inside2d
    # in-plane 경계 거리(고기 내부에서 경계까지)
    edge_dist2d = ndimage.distance_transform_edt(mask2d)

    # 슬랩으로 확장 (z 전체)
    mask = np.repeat(mask2d[None, :, :], nz, axis=0)
    edge_dist = np.repeat(edge_dist2d[None, :, :], nz, axis=0)

    # 피하지방 캡: 경계에서 rim_width 안쪽으로만(얇은 흰 테두리), 그것도 둘레의
    # 일부 구간에서만. (예전엔 각도만으로 내부 절반을 통째로 칠해서 '통살 쐐기'
    # 가 생겼다 — 캡은 반드시 경계 거리 edge_dist 로 테두리에 한정해야 한다.)
    rim_width = 4.5
    rim = np.exp(-edge_dist2d / rim_width)            # 경계서 1 → 안쪽으로 급감
    t = np.clip((0.5 + 0.5 * np.sin(ang + ph[0]) - 0.45) / 0.5, 0.0, 1.0)
    cap_perim = t * t * (3.0 - 2.0 * t)               # 둘레 일부 구간만 부드럽게
    cap2d = rim * cap_perim * mask2d
    cap_weight = np.repeat(cap2d[None, :, :], nz, axis=0)

    return {
        "mask": mask,
        "edge_dist": edge_dist.astype(np.float32),
        "cap_weight": cap_weight.astype(np.float32),
    }
