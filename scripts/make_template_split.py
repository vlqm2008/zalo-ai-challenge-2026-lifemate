#!/usr/bin/env python3
"""Template-disjoint dev split: the public test shares almost no request templates with train (top-1 kNN Jaccard
0.24 on public vs 0.61 on the stratified dev split, EXPERIMENTS 5), so a dev split must hold out whole template
clusters or its numbers are optimistic (few-shot and SFT both see the template).

  python scripts/make_template_split.py --data ./data --n-dev 300 --sim 0.5 --out-dev splits/dev2_ids.txt --out-train splits/train2_ids.txt
"""
import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.normalize_vi import norm_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--sim", type=float, default=0.5, help="link two requests into one template cluster at this Jaccard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dev", default="splits/dev2_ids.txt")
    ap.add_argument("--out-train", default="splits/train2_ids.txt")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(Path(args.data) / args.split / f"{args.split}.jsonl", encoding="utf-8") if l.strip()]
    ids = [r["id"] for r in rows]
    txt = lambda x: x.get("text", "") if isinstance(x, dict) else (x or "")  # noqa: E731
    toks = [set(norm_tokens(txt(r.get("audio_text")))) for r in rows]
    n = len(rows)
    jac = lambda a, b: len(a & b) / max(1, len(a | b))  # noqa: E731

    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if jac(toks[i], toks[j]) >= args.sim:
                parent[find(i)] = find(j)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    sizes = sorted((len(c) for c in clusters.values()), reverse=True)
    print(f"{n} samples, {len(clusters)} template clusters at Jaccard>={args.sim}; largest {sizes[:8]}; singletons {sizes.count(1)}")

    rng = random.Random(args.seed)
    order = list(clusters.values()); rng.shuffle(order)
    dev = []
    for c in order:
        if len(dev) >= args.n_dev:
            break
        dev.extend(c)
    dev_set = set(dev); train = [i for i in range(n) if i not in dev_set]
    # honesty check: top-1 similarity of each dev request to the train part (compare with public ~0.24 mean)
    top1 = [max(jac(toks[d], toks[t]) for t in train) for d in dev]
    print(f"dev {len(dev)} / train {len(train)}; dev->train top-1 Jaccard mean {st.mean(top1):.2f} median {st.median(top1):.2f}; "
          f">=0.8: {sum(x >= 0.8 for x in top1) / len(top1):.0%}; <0.4: {sum(x < 0.4 for x in top1) / len(top1):.0%}")
    Path(args.out_dev).write_text("\n".join(ids[i] for i in sorted(dev)) + "\n", encoding="utf-8")
    Path(args.out_train).write_text("\n".join(ids[i] for i in train) + "\n", encoding="utf-8")
    print("wrote", args.out_dev, args.out_train)


if __name__ == "__main__":
    main()
