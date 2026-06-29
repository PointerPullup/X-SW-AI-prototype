"""한우 1++ 등심 마블링 '지방 확률장' 청사진 생성.

이 모듈이 만드는 ``p_fat[z,y,x]`` 는 **AI 세그멘테이션이 출력했다고 가정하는**
3D 지방 확률장이다. prototype4 의 마블링다움(전면에 고르게 깔린 미세 서리꽃,
결 따라 흐르는 가는 레이스, 드문 가는 지방 줄기, 얇은 테두리 캡)은 전부 이
확률장 안에 담긴다. 시뮬레이터(``simulate.py``)는 이 확률장을 **해석만** 하며
구조를 새로 만들지 않는다.

따라서 이 파일을 실제 AI 출력으로 교체하면 파이프라인이 그대로 동작한다.

설계 원칙(실제 1++ 등심 사진 기준):
  - 마블링은 단면 **전면에 고르게** 퍼진 **미세한** 서리꽃이다(한쪽만 몰리거나
    덩어리지지 않는다). 그래서 확률장을 **고주파 우세**로 만들어, 상위
    target_fat 로 임계화했을 때 거대 저주파 덩어리 대신 촘촘한 미세 알갱이가
    전면에 고르게 남게 한다.
  - 굵은 지방 줄기(근간지방 seam)와 피하지방 캡은 **드물고 가늘게**, 베이스
    마블링 위에 ``maximum`` 으로 덧씌운다 — 더하고 재정규화하면 그 영역이 통째로
    지방으로 먹혀 비현실적 '통살 쐐기' 가 생기기 때문이다(이전 버전의 결함).

좌표 규약: ``shape=(nz, ny, nx)``
  - z = 두께(thickness, 얇음)
  - x = 등심의 긴 방향(= 근섬유 결 방향)
  - y = 등심의 짧은 방향
"""

import numpy as np

from .noise import fbm3, ridged3, make_perm, normalize01


# 기본 파라미터 — 한우 1++ 등심(BMS 8~9)의 가늘고 촘촘하며 전면에 고른
# 서리꽃 마블링을 겨냥한 값.
DEFAULTS = {
    "target_fat": 0.28,       # 확률장 상위 약 28% 가 '지방다운' 확률을 갖게(등급 손잡이)
    "prob_spread": 0.045,     # 로지스틱 매핑 부드러움(작을수록 또렷한 확률 경계)

    # 결 휘어짐(결이 x 를 따라 흐르며 y 로 완만하게 물결)
    "warp_amp": 0.05,
    "warp_freq": 0.012,

    # 결 방향 이방성: <1 이면 x(결) 방향으로 패턴이 길쭉해진다.
    "fiber_aniso": 0.32,

    # (a) 미세 서리꽃 — 고주파 점박이(**우세 성분**, 전면에 고르게 분포).
    #     결 방향으로 살짝 늘여(이방성) 사진처럼 흐름이 보이게.
    "fine_freq": 0.32,
    "fine_aniso": 0.60,
    "fine_sharpness": 1.7,
    "w_fine": 0.60,

    # (b) 가는 레이스 — 결 따라 흐르는 가는 지방 선(방향성의 주역). ridged 능선.
    "net_freq": 0.135,
    "net_sharpness": 1.45,
    "w_net": 0.55,

    # (c) 드문 가는 지방 줄기(근간지방 seam) — 저주파 능선의 '날카로운 꼭대기'만
    #     가늘게 남기고, 다시 저주파 게이트로 둘레 일부 구역에만 나타나게.
    "seam_freq": 0.030,
    "seam_aniso": 0.22,
    "seam_lo": 0.80, "seam_hi": 0.95,     # 능선 꼭대기만 통과 → 줄기를 가늘게
    "gate_freq": 0.018,
    "gate_lo": 0.50, "gate_hi": 0.78,     # 일부 구역에만 줄기 출현(드문드문)

    # 합성/오버레이 레벨
    "base_level": 0.90,       # 베이스 마블링 상한(줄기/캡이 그 위로 솟게)
    "seam_level": 1.0,        # 줄기 오버레이 강도
    "cap_strength": 0.97,     # 얇은 테두리 캡(maximum 오버레이; 안쪽을 침범하지 않음)
}


def _logistic(x, x0, spread):
    return 1.0 / (1.0 + np.exp(-(x - x0) / max(float(spread), 1e-6)))


def _smoothstep(x, edge0, edge1):
    """edge0~edge1 구간을 0→1 로 부드럽게 통과시키는 스무스스텝."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fat_probability_blueprint(shape, mask, params=None, seed=20260629,
                              cap_weight=None):
    """사실적 1++ 마블링 지방 확률장 ``p_fat[z,y,x]`` 를 만든다.

    Parameters
    ----------
    shape : (nz, ny, nx)
    mask  : (nz, ny, nx) bool — 고기(tissue) 영역.
    params: DEFAULTS 를 덮어쓸 dict.
    cap_weight : (nz,ny,nx) [0,1] — 둘레 **얇은 테두리** 피하지방 캡 가중
                 (steak_shape 출력). 0 인 안쪽은 캡 영향 없음.
    """
    nz, ny, nx = shape
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    mask = mask.astype(bool)

    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    zz = zz.astype(np.float64)
    yy = yy.astype(np.float64)
    xx = xx.astype(np.float64)

    # --- 1. 결 휘어짐: x 를 따라 흐르는 결이 y 로 완만하게 물결치게 ---
    warp = fbm3(xx, yy, zz * 0.6, make_perm(seed + 1),
                octaves=3, base_freq=p["warp_freq"])
    yw = yy + p["warp_amp"] * ny * warp        # 결 휘어짐 반영한 y'

    ax = p["fiber_aniso"]                       # x(결) 방향 압축 → 길쭉

    # --- 2. 미세 서리꽃(고주파 점박이, 전면 고름) — 우세 성분 ---
    fine = fbm3(xx * p["fine_aniso"], yw, zz, make_perm(seed + 3),
                octaves=4, base_freq=p["fine_freq"])
    fine = normalize01(fine) ** p["fine_sharpness"]

    # --- 3. 가는 레이스(결 따라 흐르는 가는 지방 선) — ridged 능선 ---
    net = ridged3(xx * ax, yw, zz, make_perm(seed + 2),
                  octaves=4, base_freq=p["net_freq"])
    net = normalize01(net) ** p["net_sharpness"]

    # --- 4. 고른 베이스 마블링(고주파 우세) ---
    #   고주파가 우세하므로 상위 분위수로 잘라도 거대 덩어리 대신 미세 알갱이가
    #   전면에 고르게 남는다.
    base = normalize01(p["w_fine"] * fine + p["w_net"] * net)

    # --- 5. 드문 가는 지방 줄기(근간지방 seam) ---
    #   저주파 능선의 날카로운 꼭대기만(가늘게) + 저주파 게이트로 일부 구역에만.
    seam = normalize01(ridged3(xx * p["seam_aniso"], yw, zz, make_perm(seed + 4),
                               octaves=3, base_freq=p["seam_freq"]))
    seam = _smoothstep(seam, p["seam_lo"], p["seam_hi"])
    gate = normalize01(fbm3(xx * 0.5, yw, zz, make_perm(seed + 5),
                            octaves=2, base_freq=p["gate_freq"]))
    seam = seam * _smoothstep(gate, p["gate_lo"], p["gate_hi"])

    # --- 6. 합성: 고른 베이스 위에 줄기/캡을 maximum 으로 덧씌움 ---
    #   (더하고 재정규화하면 그 영역이 통째로 지방으로 먹혀 '통살 쐐기' 가 생긴다.
    #    maximum 은 가는 줄기/얇은 테두리만 보장 지방으로 올리고 주변은 보존.)
    density = np.maximum(base * p["base_level"], p["seam_level"] * seam)

    if cap_weight is not None and p["cap_strength"] > 0:
        density = np.maximum(density, p["cap_strength"] * cap_weight)

    density = normalize01(density * mask)

    # --- 7. 확률로 변환: 상위 target_fat 정도가 '지방다운 확률'을 갖게 ---
    vals = density[mask]
    if vals.size == 0:
        return np.zeros(shape, dtype=np.float32)
    cut = float(np.quantile(vals, 1.0 - float(np.clip(p["target_fat"], 0.02, 0.9))))
    p_fat = _logistic(density, cut, p["prob_spread"]) * mask
    return p_fat.astype(np.float32)
