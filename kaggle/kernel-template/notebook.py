# Template: copy this folder per experiment, change "id"/"title" in kernel-metadata.json.
# Push with: kaggle kernels push -p kaggle/<folder>
from collections import defaultdict
from pathlib import Path

INPUT = Path("/kaggle/input")

# Group files by their parent dir so attached datasets are easy to spot regardless of mount depth.
dirs = defaultdict(lambda: [0, 0.0])
for p in INPUT.rglob("*"):
    if p.is_file():
        dirs[p.parent][0] += 1
        dirs[p.parent][1] += p.stat().st_size / 1e6
for d, (n, mb) in sorted(dirs.items())[:60]:
    print(f"{n:5d} files {mb:9.1f} MB  {d}")
