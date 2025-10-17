#Convert Multi_class to Binary
import os
import nibabel as nib
import numpy as np
import json
from pathlib import Path

# Paths
base_dir = "/content/drive/MyDrive/MRA"
#base_dir = "/content/MambaVesselNet_dataset_MRA_multiclass"

split_folders = {
    "training": ("imagesTr", "labelsTr"),
    "validation": ("imagesVal", "labelsVal"),
    "test": ("imagesTs", "labelsTs")
}

# Output folders for binary masks
output_base = base_dir + "_binary"
os.makedirs(output_base, exist_ok=True)

new_json = {
    "name": "MRA Vessel Segmentation Binary",
    "tensorImageSize": "3D",
    "modality": {"0": "MR"},
    "labels": {"0": "Background", "1": "Vessel"},
    "numTraining": 0,
    "numValidation": 0,
    "training": [],
    "validation": [],
    "test": []
}

for split, (img_folder, lbl_folder) in split_folders.items():
    img_path = Path(base_dir) / img_folder
    lbl_path = Path(base_dir) / lbl_folder

    out_img_path = Path(output_base) / img_folder
    out_lbl_path = Path(output_base) / lbl_folder
    out_img_path.mkdir(parents=True, exist_ok=True)
    out_lbl_path.mkdir(parents=True, exist_ok=True)

    for img_file in img_path.glob("*_0000.nii"):
        basename = img_file.stem.replace("_0000", "")
        lbl_file = lbl_path / f"{basename}.nii"

        # Load mask
        mask = nib.load(lbl_file).get_fdata()
        binary_mask = (mask > 0).astype(np.uint8)  # Convert all vessels to 1

        # Save binary mask
        nib.save(nib.Nifti1Image(binary_mask, affine=nib.load(lbl_file).affine),
                 out_lbl_path / f"{basename}.nii")

        # Copy image path
        os.symlink(img_file, out_img_path / img_file.name)  # keeps the same image

        # Add to JSON
        entry = {
            "image": str(out_img_path / img_file.name),
            "label": str(out_lbl_path / f"{basename}.nii")
        }
        if split == "training":
            new_json["training"].append(entry)
        elif split == "validation":
            new_json["validation"].append(entry)
        else:
            new_json["test"].append(entry)

# Update counts
new_json["numTraining"] = len(new_json["training"])
new_json["numValidation"] = len(new_json["validation"])

# Save JSON
json_file = Path(output_base) / "dataset.json"
with open(json_file, "w") as f:
    json.dump(new_json, f, indent=4)

print("Binary masks created and JSON saved at:", json_file)
