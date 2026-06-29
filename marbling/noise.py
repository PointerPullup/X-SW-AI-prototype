"""
noise.py
========
기초 procedural noise 알고리즘 모음.

이 모듈은 마블링 시뮬레이션의 "유기적 불규칙성"을 책임지는 가장 밑단의 도구다.
초안 PDF의 6-6 (Perlin Noise) 과 클로드의견 PDF 의 다음 항목들을 구현한다.
  - fBm (다중 옥타브 Perlin) ............ 도메인 워핑/임계값 변조의 재료
  - Ridged multifractal (abs noise 누적) . 가는 능선(잔가지) 텍스처

모든 함수는 numpy 배열 전체에 대해 vectorize 되어 있어서, 수백만 개의 voxel
좌표를 한 번에 평가할 수 있다 (파이썬 루프 없음 = 큰 grid 에서도 실용적).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Perlin gradient noise (3D, vectorized)
# ---------------------------------------------------------------------------
# Perlin noise 의 핵심: 정수 격자점마다 "임의의 gradient 벡터"를 부여하고,
# 임의의 좌표에서는 주변 8개 격자점의 gradient·거리 내적을 부드럽게 보간한다.
# 가까운 좌표끼리는 비슷한 값, 멀어질수록 부드럽게 변하는 성질을 가진다.

# 12개의 표준 3D gradient 방향 (정육면체 모서리 방향). Perlin 의 ref 구현과 동일.
_GRAD3 = np.array(
    [
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    ],
    dtype=np.float64,
)


def make_perm(seed: int) -> np.ndarray:
    """seed 로부터 512 길이의 permutation 테이블을 만든다.

    0..255 를 섞은 뒤 두 번 이어 붙여(512) 인덱스 wrap-around 시 modulo 연산을
    생략할 수 있게 한다 (고전적인 Perlin 트릭)."""
    rng = np.random.default_rng(seed)
    p = np.arange(256, dtype=np.int32)
    rng.shuffle(p)
    return np.concatenate([p, p]).astype(np.int32)


def _fade(t):
    """Perlin 의 5차 보간 곡선 6t^5 - 15t^4 + 10t^3.
    1차/3차 보간보다 격자 경계에서 미분이 매끄러워 격자 무늬(artifact)가 적다."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a, b, t):
    return a + t * (b - a)


def perlin3(x, y, z, perm):
    """3D Perlin noise. 입력 x,y,z 는 같은 shape 의 numpy 배열.
    반환값은 대략 [-1, 1] 범위."""
    xi = np.floor(x).astype(np.int32) & 255
    yi = np.floor(y).astype(np.int32) & 255
    zi = np.floor(z).astype(np.int32) & 255

    xf = x - np.floor(x)
    yf = y - np.floor(y)
    zf = z - np.floor(z)

    u, v, w = _fade(xf), _fade(yf), _fade(zf)

    def grad(ix, iy, iz, fx, fy, fz):
        # 세 번 중첩된 permutation 으로 격자점 (ix,iy,iz) 의 gradient 인덱스를 해시.
        h = perm[(perm[(perm[ix & 511] + iy) & 511] + iz) & 511] % 12
        g = _GRAD3[h]
        return g[..., 0] * fx + g[..., 1] * fy + g[..., 2] * fz

    # 정육면체 8개 모서리에서의 gradient·거리 내적
    n000 = grad(xi,     yi,     zi,     xf,     yf,     zf)
    n100 = grad(xi + 1, yi,     zi,     xf - 1, yf,     zf)
    n010 = grad(xi,     yi + 1, zi,     xf,     yf - 1, zf)
    n110 = grad(xi + 1, yi + 1, zi,     xf - 1, yf - 1, zf)
    n001 = grad(xi,     yi,     zi + 1, xf,     yf,     zf - 1)
    n101 = grad(xi + 1, yi,     zi + 1, xf - 1, yf,     zf - 1)
    n011 = grad(xi,     yi + 1, zi + 1, xf,     yf - 1, zf - 1)
    n111 = grad(xi + 1, yi + 1, zi + 1, xf - 1, yf - 1, zf - 1)

    # x → y → z 순서로 3선형 보간
    x00 = _lerp(n000, n100, u)
    x10 = _lerp(n010, n110, u)
    x01 = _lerp(n001, n101, u)
    x11 = _lerp(n011, n111, u)
    y0 = _lerp(x00, x10, v)
    y1 = _lerp(x01, x11, v)
    return _lerp(y0, y1, w)


# ---------------------------------------------------------------------------
# fBm — fractional Brownian motion (다중 옥타브 Perlin)
# ---------------------------------------------------------------------------
def fbm3(x, y, z, perm, octaves=5, lacunarity=2.0, gain=0.5, base_freq=1.0):
    """여러 주파수(octave)의 Perlin 을 누적해서 디테일이 풍부한 noise 를 만든다.

    - lacunarity: octave 마다 주파수를 곱하는 배수 (보통 2.0 → 한 옥타브 위)
    - gain      : octave 마다 진폭을 곱하는 배수 (보통 0.5 → 점점 약하게)
    굵은 형태(저주파)와 잔디테일(고주파)이 함께 들어간다. 도메인 워핑과
    임계값 변조의 기본 재료로 쓴다.  반환값은 대략 [-1,1] 로 정규화."""
    total = np.zeros_like(x, dtype=np.float64)
    freq = base_freq
    amp = 1.0
    norm = 0.0
    for _ in range(octaves):
        total += amp * perlin3(x * freq, y * freq, z * freq, perm)
        norm += amp
        freq *= lacunarity
        amp *= gain
    return total / max(norm, 1e-9)


# ---------------------------------------------------------------------------
# Ridged multifractal — 가는 능선(잔가지) 텍스처
# ---------------------------------------------------------------------------
def ridged3(x, y, z, perm, octaves=5, lacunarity=2.0, gain=0.5, base_freq=1.0):
    """1 - |noise| 를 octave 누적. noise 가 0 을 지나는 곳에서 값이 1로 솟구쳐
    '능선(ridge)'을 만든다. 마블링의 가늘고 날카로운 실 같은 지방 줄기를
    아주 싸게(저비용으로) 추가할 수 있다 (클로드의견의 '릿지드 멀티프랙탈').

    반환값은 [0,1] 범위. 1에 가까울수록 능선(=잔가지 후보)."""
    total = np.zeros_like(x, dtype=np.float64)
    freq = base_freq
    amp = 1.0
    norm = 0.0
    prev = 1.0
    for _ in range(octaves):
        n = perlin3(x * freq, y * freq, z * freq, perm)
        signal = 1.0 - np.abs(n)      # 능선
        signal = signal * signal      # 능선을 더 날카롭게
        signal = signal * prev        # 이전 옥타브로 가중 (multifractal 특성)
        total += signal * amp
        norm += amp
        prev = np.clip(signal, 0.0, 1.0)
        freq *= lacunarity
        amp *= gain
    return total / max(norm, 1e-9)


def normalize01(a):
    """배열을 [0,1] 로 min-max 정규화 (시각화/합성 편의용)."""
    lo, hi = np.min(a), np.max(a)
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)
