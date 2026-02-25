import csv

from PIL import Image
from ultralytics import SAM
import numpy as np
import cv2 as cv
import os



def image_padding():
    files = sorted(os.scandir("preprocessed/sharpened"),
                   key=lambda f: int(f.name.split("_")[1].split(".")[0]))
    for index, image in enumerate(files):
        image = Image.open(image.path)
        w, h = image.size

        max_side = max(w, h)

        # 2. Create new square image with black background
        new_img = Image.new("RGB", (max_side, max_side), (0, 0, 0))

        # 3. Compute top-left corner to center the feather
        x = (max_side - w) // 2
        y = (max_side - h) // 2

        # 4. Paste feather into square canvas
        new_img.paste(image, (x, y))

        # 5. Resize to final training resolution
        new_img = new_img.resize((224, 224), Image.LANCZOS)

        # Save
        output_path = os.path.join("preprocessed/padding", f"padded_{index}.png")
        new_img.save(output_path)

    print("Padding and resizing complete!")



def clean_mask(mask, close_kernel=5, min_component_area=500):
    """
    mask: uint8 0/1 numpy array
    close_kernel: kernel size for morphological closing (to fill holes)
    min_component_area: remove connected components smaller than this (pixels)
    """
    if close_kernel and close_kernel > 0:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_CLOSE, k)

    # remove small components
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for lab in range(1, num_labels):
        area = stats[lab, cv.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == lab] = 1
    return cleaned

def segmentation(input_dir="img",
                 output_dir="preprocessed/segmentation",
                 sam_weights_path="sam3.pt",
                 save_all_masks=False,
                 close_kernel=5,
                 min_component_area=500,
                 stats_csv_path="preprocessed/segmentation/segmentation_stats.csv",
                 thresholds=None,
                 verbose=True):
    """
    Improved segmentation pipeline that computes mask/bbox/image ratios and optionally filters segments.

    thresholds: dict with optional keys:
        - min_mask_to_bbox: e.g. 0.15    # mask must occupy >= 15% of bbox
        - max_black_ratio:    0.6       # black_in_bbox / bbox_area must be <= 0.6
        - min_mask_to_image:  0.0005    # mask must be >= 0.05% of full image
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(stats_csv_path), exist_ok=True)

    if thresholds is None:
        thresholds = {}

    # Load SAM once
    model = SAM(sam_weights_path)
    model.to("cuda")

    saved = []
    rows = []
    img_index = 0

    for entry in os.scandir(input_dir):
        if not entry.is_file():
            continue
        image_path = entry.path
        image = cv.imread(image_path)
        if image is None:
            if verbose:
                print("Could not read", image_path)
            continue
        h, w = image.shape[:2]

        # Run SAM for this image
        results = model(image_path, save=False, device=0)  # change args to match your SAM wrapper
        # results[0].masks.data is assumed as in your snippet

        for mask_i, mask_t in enumerate(results[0].masks.data):
            mask = mask_t.cpu().numpy().astype(np.uint8)  # 0/1
            # resize mask to original image size (SAM sometimes returns lower res)
            mask = cv.resize(mask, (w, h), interpolation=cv.INTER_NEAREST)
            # clean mask a little
            mask = clean_mask(mask, close_kernel=close_kernel, min_component_area=min_component_area)

            ys, xs = np.where(mask == 1)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            # Add +1 to include last pixel
            bbox_w = x2 - x1 + 1
            bbox_h = y2 - y1 + 1
            bbox_area = float(bbox_w * bbox_h)

            mask_area = float(mask.sum())
            mask_to_bbox_ratio = mask_area / bbox_area if bbox_area > 0 else 0.0
            mask_to_image_ratio = mask_area / float(w * h)
            black_in_bbox = bbox_area - mask_area
            black_ratio = black_in_bbox / bbox_area if bbox_area > 0 else 1.0

            # contours / shape info
            contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            num_contours = len(contours)
            bbox_aspect_ratio = float(bbox_w) / float(bbox_h) if bbox_h > 0 else 0.0

            # prepare cropped segmented image (object on black background)
            mask_3ch = np.stack([mask] * 3, axis=-1)
            segmented = image * mask_3ch
            cropped = segmented[y1:y2+1, x1:x2+1]  # include endpoints

            # diagnostics: overlay bbox + metrics onto a copy of original for quick manual inspection
            vis = image.copy()
            cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            txt = f"m/b:{mask_to_bbox_ratio:.2f} m/i:{mask_to_image_ratio:.4f} br:{black_ratio:.2f}"
            cv.putText(vis, txt, (x1, max(10, y1-6)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv.LINE_AA)

            # decide whether to accept this mask based on thresholds (if provided)
            accept = True
            reason_reject = ""
            if "min_mask_to_bbox" in thresholds and mask_to_bbox_ratio < thresholds["min_mask_to_bbox"]:
                accept = False
                reason_reject = f"mask_to_bbox {mask_to_bbox_ratio:.3f} < min {thresholds['min_mask_to_bbox']}"
            if "max_black_ratio" in thresholds and black_ratio > thresholds["max_black_ratio"]:
                accept = False
                reason_reject = f"black_ratio {black_ratio:.3f} > max {thresholds['max_black_ratio']}"
            if "min_mask_to_image" in thresholds and mask_to_image_ratio < thresholds["min_mask_to_image"]:
                accept = False
                reason_reject = f"mask_to_image {mask_to_image_ratio:.6f} < min {thresholds['min_mask_to_image']}"

            # Save stats row
            row = {
                "image_path": image_path,
                "mask_index": mask_i,
                "mask_area": int(mask_area),
                "bbox_area": int(bbox_area),
                "mask_to_bbox": mask_to_bbox_ratio,
                "mask_to_image": mask_to_image_ratio,
                "black_in_bbox": int(black_in_bbox),
                "black_ratio": black_ratio,
                "bbox_aspect": bbox_aspect_ratio,
                "num_contours": num_contours,
                "accepted": accept,
                "reject_reason": reason_reject
            }
            rows.append(row)

            # Save files if accepted (or optionally save all masks for inspection)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cropped_fname = os.path.join(output_dir, f"{base_name}_mask{mask_i}_crop.png")
            # vis_fname = os.path.join(output_dir, f"{base_name}_mask{mask_i}_vis.png")
            # mask_fname = os.path.join(output_dir, f"{base_name}_mask{mask_i}_mask.png")

            # always save mask and visual for inspection if save_all_masks True
            if save_all_masks or accept:
                cv.imwrite(cropped_fname, cropped)
                # cv.imwrite(vis_fname, vis)
                # cv.imwrite(mask_fname, (mask * 255).astype(np.uint8))
            if accept:
                saved.append(cropped_fname)

            if verbose:
                print(f"[{img_index}] {base_name} mask#{mask_i}: mask_area={int(mask_area)}, "
                      f"mask/bbox={mask_to_bbox_ratio:.3f}, mask/img={mask_to_image_ratio:.6f}, "
                      f"black_ratio={black_ratio:.3f} -> {'ACCEPT' if accept else 'REJECT'} {reason_reject}")

        img_index += 1

    # write CSV summary
    fieldnames = ["image_path", "mask_index", "mask_area", "bbox_area",
                  "mask_to_bbox", "mask_to_image", "black_in_bbox", "black_ratio",
                  "bbox_aspect", "num_contours", "accepted", "reject_reason"]
    with open(stats_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if verbose:
        print(f"Saved {len(saved)} accepted crops. Stats CSV: {stats_csv_path}")

    return saved, stats_csv_path

def denoise():

    for index ,image in enumerate (os.scandir("preprocessed/segmentation")):
        img = cv.imread(image.path)
        dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        filename = os.path.join("preprocessed/denoise", f"denoise_{index}.png")
        cv.imwrite(filename, dst)

def enhance_contrast():
    files = sorted(os.scandir("preprocessed/denoise"),
                   key=lambda f: int(f.name.split("_")[1].split(".")[0]))

    for index,image in enumerate (files):
        img = cv.imread(image.path)
        # Convert to LAB color space
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

        # Split channels
        l, a, b = cv.split(lab)

        # Apply CLAHE only to the L channel
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge enhanced L with original A and B
        lab_enhanced = cv.merge((cl, a, b))

        # Convert back to BGR
        enhanced = cv.cvtColor(lab_enhanced, cv.COLOR_LAB2BGR)

        # Save result
        filename = os.path.join("preprocessed/contrast", f"contrast_{index}.png")
        cv.imwrite(filename, enhanced)

def sharpening(strength: float = 0.9):
    files = sorted(os.scandir("preprocessed/contrast"),
                   key=lambda f: int(f.name.split("_")[1].split(".")[0]))

    for index, image in enumerate(files):
        img = cv.imread(image.path)
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * strength

        sharpened = cv.filter2D(img, -1, kernel)
        filename = os.path.join("preprocessed/sharpened", f"sharpening_{index}.png")
        cv.imwrite(filename, sharpened)

def data_preprocess(dir):
    thresholds = {
    "min_mask_to_image": 0.20,
    "min_mask_to_bbox": 0.30
    }
    segmentation(input_dir=dir, save_all_masks=False, thresholds=thresholds)
    denoise()
    enhance_contrast()
    sharpening()
    image_padding()

