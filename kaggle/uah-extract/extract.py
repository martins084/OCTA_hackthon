# Copy only the sensor/text files of UAH-DriveSet (drop the videos) into one small zip.
import shutil
from pathlib import Path

SRC = Path("/kaggle/input/datasets/jaymovaliya14/uah-driveset/UAH-DRIVESET-v1")
OUT = Path("/kaggle/working/uah")
for p in SRC.glob("D*/*/*.txt"):
    dst = OUT / p.relative_to(SRC)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, dst)
shutil.make_archive("/kaggle/working/uah_txt", "zip", OUT)
shutil.rmtree(OUT)
print("trips:", len(list(SRC.glob("D*/*/"))))
