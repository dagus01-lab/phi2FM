from huggingface_hub import whoami, snapshot_download

print("HF user:", whoami()["name"])
target = "/Users/jja/Documents/02_EOFM/phi2FM/downstream/data"
files = snapshot_download(
  repo_id="ESA-PhiLab-Edge/OEOBench-Burnt_Area_Dataset",
  repo_type="dataset",
  revision="main",
  local_dir=target,
  local_dir_use_symlinks=False,
  # Narrow the download if you want only specific folders/files:
  # allow_patterns=["cloudsen12/**", "README.md", "dataset_card.json"]
)
print(f"Downloaded {len(files)} files to {target}")