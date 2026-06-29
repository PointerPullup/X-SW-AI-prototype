"""구조텐서(structure tensor) 기반 근섬유 결 방향 추정.

마블링이 결을 따라 길쭉하면, 지방 확률장의 밝기 변화(gradient)는 결에
**수직**으로 가장 강하다. 각 단면에서 구조텐서를 만들어 그 주축을 구하면,
주축에 수직인 방향이 곧 근섬유 결 방향이 된다.

이 추정은 **오직 입력 확률장 ``p_fat`` 에서만** 나온다(절차적 가정 없음).
확률장을 바꾸면 추정 결 방향도 따라 바뀐다.

반환 결 방향은 면내(x-y) 단위벡터다. z(두께)는 결의 주 평면이 아니므로
면외 성분 vz ≈ 0 으로 둔다(바이오프린팅 정렬 벡터와 일관).
"""

import numpy as np
from scipy import ndimage


def estimate_fiber_orientation(p_fat, mask, sigma_grad=1.1, sigma_tensor=4.5):
    """확률장에서 복셀별 결 방향(dy, dx)과 방향 일관성(coherence)을 추정.

    Parameters
    ----------
    p_fat : (nz, ny, nx) float — 지방 확률장.
    mask  : (nz, ny, nx) bool  — 고기 영역.
    sigma_grad   : gradient 전 평활(노이즈 억제).
    sigma_tensor : 구조텐서 적분 스케일(클수록 더 광역적인 결).

    Returns
    -------
    dict: dy, dx (면내 단위벡터), coherence (0~1), angle (라디안, atan2(dy,dx)).
    """
    nz, ny, nx = p_fat.shape
    mask = mask.astype(bool)
    dy = np.zeros((nz, ny, nx), dtype=np.float32)
    dx = np.zeros((nz, ny, nx), dtype=np.float32)
    coh = np.zeros((nz, ny, nx), dtype=np.float32)

    for z in range(nz):
        m = mask[z]
        if not m.any():
            continue
        f = ndimage.gaussian_filter(p_fat[z].astype(np.float64), sigma_grad)
        gy, gx = np.gradient(f)                      # [d/dy, d/dx]

        # 구조텐서 성분(가우시안으로 국소 적분).
        Jyy = ndimage.gaussian_filter(gy * gy, sigma_tensor)
        Jxx = ndimage.gaussian_filter(gx * gx, sigma_tensor)
        Jxy = ndimage.gaussian_filter(gx * gy, sigma_tensor)

        # 2x2 대칭행렬 고유분해(closed form).
        # 주축(최대 고유값) 각도 θ = ½·atan2(2Jxy, Jxx−Jyy) — gradient 강한 방향
        # = 마블링을 가로지르는 방향. 결은 그에 수직: (−sinθ, cosθ).
        theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
        fdx = -np.sin(theta)
        fdy = np.cos(theta)

        # 방향 일관성: (λmax−λmin)/(λmax+λmin). 1=뚜렷한 결, 0=등방(점박이).
        tmp = np.sqrt(np.maximum((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2, 0.0))
        tr = Jxx + Jyy
        coherence = np.where(tr > 1e-12, tmp / (tr + 1e-12), 0.0)

        dy[z] = (fdy * m).astype(np.float32)
        dx[z] = (fdx * m).astype(np.float32)
        coh[z] = (np.clip(coherence, 0.0, 1.0) * m).astype(np.float32)

    angle = np.arctan2(dy, dx)
    return {"dy": dy, "dx": dx, "coherence": coh, "angle": angle}
