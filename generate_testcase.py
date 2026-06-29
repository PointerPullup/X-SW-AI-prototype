"""prototype4 테스트케이스 생성 — 사실적 한우 1++ 지방 확률장(청사진).

이 스크립트가 만드는 ``p_fat[z,y,x]`` 는 **세그멘테이션 AI 가 출력했다고 가정하는**
3D 지방 확률장이다. 마블링의 모든 구조(결 따라 길쭉한 그물 · 미세 서리꽃 ·
굵은 지방 줄기 · 피하지방 캡)가 이 확률장 안에 담긴다.

simulate.py 는 이 확률장만 보고 마블링을 만든다. 즉 이 파일을 실제 AI 출력으로
교체하면 파이프라인이 그대로 동작한다.

사용:
    python generate_testcase.py --out testcase.npz
"""

import argparse
import os

import numpy as np

from marbling.blueprint import fat_probability_blueprint, DEFAULTS
from marbling.steak_shape import loin_slab_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nz", type=int, default=56, help="두께")
    ap.add_argument("--ny", type=int, default=340, help="등심 짧은 방향")
    ap.add_argument("--nx", type=int, default=420, help="등심 긴 방향(결)")
    ap.add_argument("--target-fat", type=float, default=0.28,
                    help="확률장 상위 분위수 기준 지방 비율(등급 손잡이)")
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--out", default="testcase.npz")
    args = ap.parse_args()

    shape = (args.nz, args.ny, args.nx)
    print(f"[testcase] generating fat-probability blueprint {shape} "
          f"({np.prod(shape):,} voxels) ...")

    sm = loin_slab_mask(shape, seed=args.seed)
    mask = sm["mask"]

    params = dict(DEFAULTS)
    params["target_fat"] = args.target_fat
    p_fat = fat_probability_blueprint(shape, mask, params, seed=args.seed,
                                      cap_weight=sm.get("cap_weight"))
    p_fat = (p_fat * mask).astype(np.float32)

    np.savez_compressed(
        args.out,
        shape=np.array(shape),
        p_fat=p_fat,
        mask=mask,
        edge_dist=sm["edge_dist"],
        cap_weight=sm["cap_weight"],
    )
    tv = mask.sum()
    print(f"[testcase] loin steak mask applied "
          f"(tissue {tv:,} / {mask.size:,} = {mask.mean()*100:.1f}%)")
    print(f"[testcase] saved -> {os.path.abspath(args.out)}")
    print(f"[testcase] p_fat mean(tissue)={p_fat[mask].mean():.3f}  "
          f">0.5 voxels = {(p_fat[mask] > 0.5).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
