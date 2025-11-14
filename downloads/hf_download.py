from huggingface_hub import whoami, snapshot_download
import os

repos= ["ESA-PhiLab-Edge/OEOBench-Burnt_Area_Dataset", "ESA-PhiLab-Edge/OEOBench-WorldFloods", "ESA-PhiLab-Edge/OEOBench-AcquaAnom", "ESA-PhiLab-Edge/OEOBench-BurnScape/tree/main"] #"ESA-Phi$
print("HF user:", whoami()["name"])
target_base_dir = "/lustre/projects/1001/gdaga/home"
targets = ["burned_area.zarr", "worldfloods.zarr", "anomaly_detection.zarr", "fire.zarr"]
for i in range(len(repos)):
  target = f"{target_base_dir}/{targets[i]}"

  target_dir = os.path.join(target_base_dir, targets[i])
  print(f"Downloading {repos[i]} to {target_dir}...")
  files = snapshot_download(
    repo_id=repos[i],
    repo_type="dataset",
    revision="main",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    # Narrow the download if you want only specific folders/files:
    # allow_patterns=["cloudsen12/**", "README.md", "dataset_card.json"]
  )
  print(f"Downloaded {len(files)} files to {target}")