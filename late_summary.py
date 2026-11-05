"""Aggregate results/late_rotation/summary.json per attack."""
import collections
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'results/late_rotation/summary.json'
with open(path) as f:
    summary = json.load(f)
rows = summary['detection']
threshold = rows[0]['threshold']

print(f"threshold T >= {threshold:.2f} (fpr {summary['args']['fpr']}, union bound over 1080 angles x 65536 nonces)")
quality = summary['quality']
for key in ('psnr', 'ssim'):
    values = [q[key] for q in quality]
    print(f"{key.upper()}: mean {sum(values) / len(values):.3f}, min {min(values):.3f}, max {max(values):.3f}")

groups = collections.defaultdict(list)
for row in rows:
    groups[(row['label'], row['interpolation'], row['attack_angle'])].append(row)

print(f"\n{'label':5s} {'interp':8s} {'angle':>6s} | detected | T min / mean / max  | coarse min | angle err max | nonce ok")
for (label, interp, angle), group in sorted(groups.items(), key=lambda kv: (kv[0][0] != 'wm', kv[0][1], kv[0][2])):
    scores = [r['score'] for r in group]
    coarse = [r['coarse_score'] for r in group]
    errors = [abs((r['angle'] - angle + 180) % 360 - 180) for r in group]
    nonce_ok = sum(r['nonce_correct'] for r in group)
    detected = sum(r['detected'] for r in group)
    print(f"{label:5s} {interp:8s} {angle:>6g} | {detected:>3d}/{len(group):<3d}  | "
          f"{min(scores):5.2f} / {sum(scores) / len(scores):5.2f} / {max(scores):5.2f} | {min(coarse):10.2f} | "
          f"{max(errors) if label == 'wm' else float('nan'):13.1f} | {nonce_ok if label == 'wm' else '-'}")

wm = [r for r in rows if r['label'] == 'wm']
nowm = [r for r in rows if r['label'] == 'nowm']
print(f"\nwatermarked: {sum(r['detected'] for r in wm)}/{len(wm)} detected, min T {min(r['score'] for r in wm):.2f}, "
      f"angle and nonce correct {sum(r['nonce_correct'] for r in wm)}/{len(wm)}")
print(f"unwatermarked: {sum(r['detected'] for r in nowm)}/{len(nowm)} detected, max T {max(r['score'] for r in nowm):.2f}")
print(f"wrong key on all images (coarse search): max T {max(r['wrong_key_coarse_score'] for r in rows):.2f}")
non_anchor = [r for r in wm if r['attack_angle'] % 45 != 0]
print(f"coarse score at angles off the 45-degree anchors: min {min(r['coarse_score'] for r in non_anchor):.2f}")
seconds = [r['seconds'] for r in rows]
print(f"detection time: median {sorted(seconds)[len(seconds) // 2]:.0f}s")
