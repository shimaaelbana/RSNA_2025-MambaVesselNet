import json
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import datetime
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import AsDiscrete, Compose, EnsureType


def calculate_metrics(pred, true):
    TP = np.sum((pred == 1) & (true == 1))
    FP = np.sum((pred == 1) & (true == 0))
    TN = np.sum((pred == 0) & (true == 0))
    FN = np.sum((pred == 0) & (true == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Vp = np.sum(pred == 1)
    Vt = np.sum(true == 1)
    volume_similarity = (
        1 - abs(Vp - Vt) / (Vp + Vt - abs(Vp - Vt))
        if (Vp + Vt - abs(Vp - Vt)) > 0
        else 0
    )

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "volume_similarity": volume_similarity,
    }


def find_prediction_file(base_name, pred_dir):
    """
    Find the actual prediction file inside the directory structure
    """
    # First check if it's a directory
    dir_path = os.path.join(pred_dir, base_name)
    
    if os.path.isdir(dir_path):
        # Look for common prediction file patterns inside the directory
        files_in_dir = os.listdir(dir_path)
        
        # Common patterns for prediction files
        possible_files = []
        for file in files_in_dir:
            if file.endswith(('.nii', '.nii.gz')):
                possible_files.append(file)
        
        if len(possible_files) == 1:
            return os.path.join(dir_path, possible_files[0])
        elif len(possible_files) > 1:
            # If multiple files, look for specific patterns
            for file in possible_files:
                if any(pattern in file.lower() for pattern in ['seg', 'prediction', 'pred', 'mask']):
                    return os.path.join(dir_path, file)
            # If no specific pattern, take the first one
            return os.path.join(dir_path, possible_files[0])
        else:
            print(f"No .nii/.nii.gz files found in directory: {dir_path}")
            return None
    else:
        # If it's not a directory, try direct file patterns
        possible_patterns = [
            f"{base_name}.nii.gz",
            f"{base_name}.nii", 
            f"{base_name}_seg.nii.gz",
            f"{base_name}_seg.nii"
        ]
        
        for pattern in possible_patterns:
            full_path = os.path.join(pred_dir, pattern)
            if os.path.exists(full_path):
                return full_path
        
        return None


def main():
    pred_dir = "/content/drive/MyDrive/Mamba/predicted_output"
    gt_dir = "/content/MambaVesselNet/mamba/MambaVesselNet_dataset_MRA_binary/labelsTs"
    dataset_json = "/content/MambaVesselNet/mamba/MambaVesselNet_dataset_binary/dataset.json"

    with open(dataset_json, "r") as f:
        test_data = json.load(f)["test"]

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    mean_iou_metric = MeanIoU(include_background=False, ignore_empty=True)

    post_transforms = Compose([EnsureType(), AsDiscrete(threshold_values=True)])

    eval_dict = {}
    total_dice, total_iou = 0, 0
    total_sensitivity, total_specificity, total_precision, total_volume_similarity = (
        0,
        0,
        0,
        0,
    )

    successful_evaluations = 0

    for i, item in enumerate(test_data):
        base_name = os.path.splitext(os.path.basename(item["image"]))[0]
        print(f"\nProcessing {i+1}/{len(test_data)}: {base_name}")
        
        # Find prediction file (handles directory structure)
        pred_image = find_prediction_file(base_name, pred_dir)
        
        # Ground truth files have .nii extension  
        gt_base = base_name.replace("_0000", "")  # Remove _0000 from image name for GT
        true_image = os.path.join(gt_dir, f"{gt_base}.nii")

        if pred_image is None:
            print(f"[Warning] No prediction file found for: {base_name}")
            continue
            
        if not os.path.exists(true_image):
            print(f"[Warning] Missing GT: {true_image}")
            continue

        print(f"  Using prediction: {pred_image}")
        print(f"  Using ground truth: {true_image}")

        try:
            sitk_pred = sitk.ReadImage(pred_image)
            sitk_true = sitk.ReadImage(true_image)
            pred_array = sitk.GetArrayFromImage(sitk_pred)
            true_array = sitk.GetArrayFromImage(sitk_true)

            if pred_array.shape != true_array.shape:
                print(f"[Error] Shape mismatch: pred {pred_array.shape} vs true {true_array.shape}")
                continue

            pred_array = post_transforms(pred_array)
            true_array = post_transforms(true_array)

            dice_metric(y_pred=pred_array[None, None], y=true_array[None, None])
            mean_iou_metric(y_pred=pred_array[None], y=true_array[None])

            dice_score = dice_metric.aggregate().item()
            dice_metric.reset()
            mean_iou_score = mean_iou_metric.aggregate().item()
            mean_iou_metric.reset()

            additional_metrics = calculate_metrics(pred_array, true_array)
            eval_dict[base_name] = {
                "DICE": dice_score,
                "Mean IoU": mean_iou_score,
                **additional_metrics,
            }
            
            print(
                f"  DICE={dice_score:.5f}, Mean IoU={mean_iou_score:.5f}, "
                f"Sens={additional_metrics['sensitivity']:.5f}, "
                f"Spec={additional_metrics['specificity']:.5f}, "
                f"Prec={additional_metrics['precision']:.5f}, "
                f"VolSim={additional_metrics['volume_similarity']:.5f}"
            )

            total_dice += dice_score
            total_iou += mean_iou_score
            total_sensitivity += additional_metrics["sensitivity"]
            total_specificity += additional_metrics["specificity"]
            total_precision += additional_metrics["precision"]
            total_volume_similarity += additional_metrics["volume_similarity"]
            successful_evaluations += 1
            
        except Exception as e:
            print(f"[Error] Failed to process {base_name}: {str(e)}")
            continue

    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Successfully evaluated: {successful_evaluations}/{len(test_data)}")

    if successful_evaluations > 0:
        mean_dice = total_dice / successful_evaluations
        mean_iou = total_iou / successful_evaluations
        mean_sensitivity = total_sensitivity / successful_evaluations
        mean_specificity = total_specificity / successful_evaluations
        mean_precision = total_precision / successful_evaluations
        mean_volume_similarity = total_volume_similarity / successful_evaluations

        print(f"\n=== RESULTS ===")
        print(f"Overall Mean DICE: {mean_dice:.5f}")
        print(f"Overall Mean IoU: {mean_iou:.5f}")
        print(f"Overall Mean Sensitivity: {mean_sensitivity:.5f}")
        print(f"Overall Mean Specificity: {mean_specificity:.5f}")
        print(f"Overall Mean Precision: {mean_precision:.5f}")
        print(f"Overall Mean Volume Similarity: {mean_volume_similarity:.5f}")

        eval_dict["Overall Mean"] = {
            "DICE": mean_dice,
            "Mean IoU": mean_iou,
            "Sensitivity": mean_sensitivity,
            "Specificity": mean_specificity,
            "Precision": mean_precision,
            "Volume Similarity": mean_volume_similarity,
        }

        df = pd.DataFrame.from_dict(eval_dict, orient="index")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H")
        csv_filename = f"/content/evaluation_segnet_{current_time}_hours.csv"
        df.to_csv(csv_filename, index=True, header=True)
        print(f"Results saved to {csv_filename}")
    else:
        print("No evaluations completed. Please check your file paths.")


if __name__ == "__main__":
    main()