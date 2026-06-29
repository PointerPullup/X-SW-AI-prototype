"""prototype4 시뮬레이터 — 지방 확률장 → 사실적 3D 마블링 해석.

**입력 확률장(p_fat)에 100% 종속한다. 입력에 없는 구조를 지어내지 않는다.**

  1) 결 방향 추정 : p_fat 구조텐서 → 복셀별 근섬유 방향 (orientation.py)
  2) 라벨링       : p_fat 을 목표 지방비율로 임계화 → 지방/근육
  3) 형태 정리    : 미세 잡음 덩어리만 제거 (구조 보존)
  4) export       : 복셀 방향벡터 + 3D 웹 뷰어 + 단면 미리보기

사용:
    python simulate.py --testcase testcase.npz --out out
    cd out/web && python -m http.server 8002
"""

import argparse
import gzip
import json
import os
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from marbling.noise import fbm3, make_perm, normalize01
from marbling.orientation import estimate_fiber_orientation
from marbling.render import threshold_to_ratio, remove_small_components

LABEL_MUSCLE, LABEL_FAT, LABEL_AIR = 0, 1, 255

# 최종 라벨에 실제로 기여하는 단계만 노출한다(확률장 → 해석 순서).
STAGE_SPECS = [
    ("input", "0. 지방 확률장 (AI 입력)", "scalar"),
    ("meat_mask", "1. 고기 마스크", "scalar"),
    ("orientation", "2. 근섬유 결 방향 (구조텐서)", "scalar"),
    ("fat_threshold", "3. 임계화 지방 (목표비율)", "scalar"),
    ("fat_clean", "4. 정리된 지방", "scalar"),
    ("final", "5. 최종 라벨 (근육/지방)", "labels"),
]


def log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def _write_gz(path, arr):
    with gzip.open(path, "wb") as f:
        f.write(np.ascontiguousarray(arr).tobytes())


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} not JSON serializable")


def run(testcase_path, out_dir, target_fat=0.28, min_size=6, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    d = np.load(testcase_path)
    shape = tuple(int(v) for v in d["shape"])
    nz, ny, nx = shape
    p_fat = d["p_fat"].astype(np.float32)
    mask = d["mask"].astype(bool) if "mask" in d.files else np.ones(shape, bool)
    mask = ndimage.binary_fill_holes(mask)
    log(f"loaded testcase {shape} ({np.prod(shape):,} voxels, "
        f"tissue {mask.mean()*100:.1f}%, p_fat mean {p_fat[mask].mean():.3f})", t0)

    # 1) 결 방향 추정 (확률장 구조텐서) — 입력에서만 도출
    fiber = estimate_fiber_orientation(p_fat, mask)
    log(f"fiber orientation estimated "
        f"(mean coherence {fiber['coherence'][mask].mean():.3f})", t0)

    # 2) 목표 지방비율로 임계화
    fat_raw, thr = threshold_to_ratio(p_fat, mask, target_fat)
    # 3) 미세 잡음 정리(구조 보존)
    fat_clean = remove_small_components(fat_raw, min_size=min_size) & mask
    log(f"threshold p_fat>={thr:.3f}  raw={fat_raw[mask].mean()*100:.2f}%  "
        f"clean={fat_clean[mask].mean()*100:.2f}%  (target {target_fat*100:.0f}%)", t0)

    labels = np.full(shape, LABEL_MUSCLE, dtype=np.uint8)
    labels[fat_clean] = LABEL_FAT
    labels[~mask] = LABEL_AIR

    # 복셀별 결 방향 벡터(int8) — 바이오프린팅 정렬용
    fiber_dir_y = np.round(np.clip(fiber["dy"], -1, 1) * 127).astype(np.int8)
    fiber_dir_x = np.round(np.clip(fiber["dx"], -1, 1) * 127).astype(np.int8)
    fiber_dir_z = np.zeros(shape, dtype=np.int8)

    out_npz = os.path.join(out_dir, "marbling_volume.npz")
    np.savez_compressed(
        out_npz,
        shape=np.array(shape),
        labels=labels,
        fat_prob=p_fat,
        fiber_dir_y_i8=fiber_dir_y,
        fiber_dir_x_i8=fiber_dir_x,
        fiber_dir_z_i8=fiber_dir_z,
    )
    log(f"saved volume -> {out_npz}", t0)

    stage_fields = {
        "input": p_fat,
        "meat_mask": mask.astype(np.float32),
        "orientation": (((fiber["angle"] + np.pi) / (2 * np.pi)) * mask).astype(np.float32),
        "fat_threshold": fat_raw.astype(np.float32),
        "fat_clean": fat_clean.astype(np.float32),
    }

    metrics = _metrics(labels, mask, fat_clean, fiber, thr)
    _save_metrics(out_dir, testcase_path, target_fat, thr, metrics)
    export_web(out_dir, shape, labels, p_fat, stage_fields, metrics, seed, t0)
    render_previews(out_dir, shape, p_fat, labels, fiber, seed, t0)
    log("DONE.", t0)


def _uniformity_cv(fat, mask):
    """단면(z중앙)을 12x12 격자로 나눠 칸별 지방% 의 변동계수(std/mean).

    작을수록 마블링이 단면 전면에 고르게 퍼졌다는 뜻(1++ 의 핵심 특성).
    """
    nz, ny, nx = fat.shape
    z = nz // 2
    fs, ms = fat[z], mask[z]
    ys = np.linspace(0, ny, 13).astype(int)
    xs = np.linspace(0, nx, 13).astype(int)
    fr = []
    for i in range(12):
        for j in range(12):
            sm = ms[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if sm.sum() > 40:
                sf = fs[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                fr.append(sf.sum() / sm.sum())
    fr = np.asarray(fr, dtype=np.float64)
    if fr.size == 0 or fr.mean() <= 1e-9:
        return 0.0
    return float(fr.std() / fr.mean())


def _metrics(labels, mask, fat_clean, fiber, thr):
    """1++ 마블링 품질 지표.

    주의: 3D 연결요소 '최대 덩어리 비율' 은 미세 마블링이라도 두께(z) 방향으로
    연결돼 퍼콜레이션으로 거의 1.0 이 되어 **품질 신호로 쓸 수 없다**. 사람이
    실제로 보는 것은 단면(2D)이므로, 단면 기준의 알갱이 수/최대 알갱이 비/두께/
    균일성으로 평가한다.
    """
    tissue = labels != LABEL_AIR
    tv = max(int(tissue.sum()), 1)
    fat = labels == LABEL_FAT
    nz = labels.shape[0]

    # --- 단면(2D) 알갱이 통계: 사람이 보는 '미세함' 척도 ---
    f4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    counts, larges = [], []
    for z in range(4, nz - 4, 4):
        fs = fat[z]
        s = int(fs.sum())
        if s < 64:
            continue
        l2, n2 = ndimage.label(fs, structure=f4)
        if n2 == 0:
            continue
        sizes = ndimage.sum(np.ones_like(fs, dtype=np.int32), l2,
                            index=np.arange(1, n2 + 1))
        counts.append(n2)
        larges.append(float(sizes.max()) / float(s))
    slice_fleck_count = float(np.mean(counts)) if counts else 0.0
    slice_largest_ratio = float(np.mean(larges)) if larges else 0.0

    # --- 지방 feature 두께(거리변환): 미세 서리꽃이면 1~2px ---
    if fat.any():
        ftk = ndimage.distance_transform_edt(fat)
        thick_avg = float(2.0 * ftk[fat].mean())     # 대략적 평균 두께(px)
        thick_p90 = float(2.0 * np.percentile(ftk[fat], 90))
    else:
        thick_avg = thick_p90 = 0.0

    coherence = float(fiber["coherence"][tissue].mean())
    return {
        # 웹 뷰어 호환 키 — 모두 '사람이 보는 단면' 기준
        "component_count": round(slice_fleck_count, 1),     # 단면 평균 알갱이 수
        "largest_component_ratio": slice_largest_ratio,     # 단면 최대 알갱이 비(작을수록 1++)
        "average_thickness_px": thick_avg,
        "orientation_coherence": coherence,
        # 기록용 상세
        "fat_ratio": float(fat.sum() / tv),
        "slice_fleck_count": slice_fleck_count,
        "slice_largest_ratio": slice_largest_ratio,
        "fat_thickness_avg_px": thick_avg,
        "fat_thickness_p90_px": thick_p90,
        "uniformity_cv": _uniformity_cv(fat, mask),
        "fiber_coherence": coherence,
        "threshold": float(thr),
    }


def _save_metrics(out_dir, testcase_path, target_fat, thr, metrics):
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "testcase": os.path.basename(testcase_path),
            "target_fat_ratio": target_fat,
            "threshold": thr,
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False, default=_json_default)


def export_web(out_dir, shape, labels, p_fat, stage_fields, metrics, seed, t0):
    """gzip raw 배열 + meta.json 을 out/web/ 뷰어(index.html)용으로 export."""
    web_dir = os.path.join(out_dir, "web")
    os.makedirs(web_dir, exist_ok=True)
    nz, ny, nx = shape

    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    base = normalize01(fbm3(zz, yy, xx, make_perm(seed + 77),
                            octaves=3, base_freq=0.05))
    grain = normalize01(fbm3(zz * 0.7, yy * 1.4, xx * 0.35,
                             make_perm(seed + 81), octaves=2, base_freq=0.075))
    shade = normalize01(0.78 * base + 0.22 * grain)

    _write_gz(os.path.join(web_dir, "labels.bin.gz"), labels.astype(np.uint8))
    _write_gz(os.path.join(web_dir, "shade.bin.gz"), (shade * 255).astype(np.uint8))
    _write_gz(os.path.join(web_dir, "fat_prob.bin.gz"),
              (np.clip(p_fat, 0, 1) * 255).astype(np.uint8))

    stages_meta = []
    for key, name, typ in STAGE_SPECS:
        if typ == "scalar":
            if key not in stage_fields:
                continue
            fn = f"stage_{key}.bin.gz"
            _write_gz(os.path.join(web_dir, fn),
                      (np.clip(stage_fields[key], 0, 1) * 255).astype(np.uint8))
            stages_meta.append({"key": key, "name": name, "type": typ, "file": fn})
        else:
            stages_meta.append({"key": key, "name": name, "type": typ,
                                "file": "labels.bin.gz"})

    # prototype4 는 seed segment 개념이 없다(확률장 직접 해석) → 빈 배열.
    with open(os.path.join(web_dir, "seeds.json"), "w", encoding="utf-8") as f:
        json.dump([], f)

    tissue = labels != LABEL_AIR
    tv = int(tissue.sum())
    meta = {
        "nz": nz, "ny": ny, "nx": nx, "order": "C",
        "labels": {"muscle": LABEL_MUSCLE, "fat": LABEL_FAT,
                   "connective": 2, "air": LABEL_AIR},
        "stages": stages_meta,
        "stats": {
            "fat_pct": float(100.0 * (labels == LABEL_FAT).sum() / max(tv, 1)),
            "connective_pct": 0.0,
            "muscle_pct": float(100.0 * (labels == LABEL_MUSCLE).sum() / max(tv, 1)),
            "tissue_voxels": tv,
            "n_seeds": 0,
        },
        "debug": {"metrics": metrics, "calibration": {}},
    }
    with open(os.path.join(web_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=_json_default)

    template = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "index.html")
    if os.path.exists(template):
        shutil.copy(template, os.path.join(web_dir, "index.html"))
    log(f"exported web -> {web_dir} (tissue {tv:,} voxels)", t0)


def render_previews(out_dir, shape, p_fat, labels, fiber, seed, t0):
    nz, ny, nx = shape
    zc = nz // 2
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(p_fat[zc], cmap="magma", origin="lower")
    axes[0].set_title("p_fat — input probability (z-mid)")
    axes[1].imshow(labels[zc] == LABEL_FAT, cmap="gray", origin="lower")
    axes[1].set_title("fat label (z-mid)")
    axes[2].imshow(p_fat.mean(axis=0), cmap="magma", origin="lower")
    axes[2].set_title("p_fat mean projection (Z)")
    for a in axes:
        a.axis("off")
    fig.suptitle("prototype4: probability blueprint -> marbling", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "preview.png"), dpi=110)
    plt.close(fig)
    log("rendered preview.png", t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testcase", default="testcase.npz")
    ap.add_argument("--out", default="out")
    ap.add_argument("--target-fat", type=float, default=0.28,
                    help="목표 지방 면적비(확률장 임계화 기준)")
    ap.add_argument("--min-size", type=int, default=6,
                    help="제거할 미세 잡음 덩어리 최대 크기(voxel)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.testcase, args.out, target_fat=args.target_fat,
        min_size=args.min_size, seed=args.seed)


if __name__ == "__main__":
    main()
