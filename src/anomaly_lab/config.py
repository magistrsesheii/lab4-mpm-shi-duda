from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
CHECKPOINTS_DIR = ROOT / "checkpoints"

MVTEC_DIR = DATA_DIR / "mvtec_ad"
CASTING_DIR = DATA_DIR / "casting"
PCB_DIR = DATA_DIR / "pcb_aoi"

IMAGE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

OBJECT_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
