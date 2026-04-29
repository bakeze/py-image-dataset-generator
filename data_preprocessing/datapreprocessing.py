import csv
import base64
import io
from PIL import Image
import numpy as np
import cv2 as cv
import os

# Base directory for preprocessed outputs (relative to project root)
_BASE_DIR = os.path.join(os.path.dirname(__file__), "preprocessed")

def _subdir(name):
    """Return the absolute path to a preprocessed sub-directory."""
    path = os.path.join(_BASE_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def image_padding():
    """
    Final step: Pad and resize all segmented images to 224x224 with black background.
    Saves to 'image_preprocessed' folder for final model training.
    """
    # Create final output directory
    final_output_dir = os.path.join(os.path.dirname(__file__), "image_preprocessed")
    os.makedirs(final_output_dir, exist_ok=True)
    
    files = sorted(os.scandir(_subdir("contrast")),
                   key=lambda f: int(f.name.split("_")[1].split(".")[0]) if "_" in f.name else 0)
    
    processed_count = 0
    for index, image_file in enumerate(files):
        try:
            image = Image.open(image_file.path)
            w, h = image.size

            max_side = max(w, h)

            # 1. Create new square image with black background
            new_img = Image.new("RGB", (max_side, max_side), (0, 0, 0))

            # 2. Compute top-left corner to center the object
            x = (max_side - w) // 2
            y = (max_side - h) // 2

            # 3. Paste object into square canvas
            new_img.paste(image, (x, y))

            # 4. Resize to final training resolution (224x224)
            new_img = new_img.resize((224, 224), Image.LANCZOS)

            # 5. Save to image_preprocessed folder (FINAL OUTPUT)
            output_filename = f"feather_{processed_count:05d}.png"
            output_path = os.path.join(final_output_dir, output_filename)
            new_img.save(output_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ⚠️  Error processing {image_file.path}: {e}")
            continue

    print(f"\n  ✓ Padding and resizing complete!")
    print(f"  📁 Final images saved to: {final_output_dir}")
    print(f"  🎯 Total processed: {processed_count} images")
    
    return processed_count


def call_sam3_replicate(image_path, api_token=None):
    """
    Perform object segmentation using advanced edge detection + contour analysis.
    Optimized to separate individual feathers from grouped images.
    
    Returns list of masks as numpy arrays (one per feather).
    """
    image = cv.imread(image_path)
    if image is None:
        return None
    
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Step 1: Apply Gaussian blur to smooth
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Step 2: High-pass filter to enhance edges
    laplacian = cv.Laplacian(blurred, cv.CV_32F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Step 3: Apply adaptive thresholding for better separation
    binary = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 21, 5)
    
    # Step 4: Morphological operations to separate contiguous objects
    kernel_small = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    
    # Distance transform to help separate touching objects
    dist_transform = cv.distanceTransform(binary, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    _, sure_fg = cv.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Watershed algorithm to separate touching feathers
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv.dilate(binary, kernel, iterations=3)
    unknown = cv.subtract(sure_bg, sure_fg)
    
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    markers = cv.watershed(image_rgb, markers)
    
    # Extract individual masks from watershed markers
    masks = []
    unique_markers = np.unique(markers)
    min_area = (h * w) * 0.002  # Minimum 0.2% of image
    max_area = (h * w) * 0.95   # Maximum 95% of image
    
    print(f"  🔍 Detected {len(unique_markers) - 2} potential feathers...")
    
    for marker_id in unique_markers:
        if marker_id <= 1:  # Skip background (0) and border (1)
            continue
        
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[markers == marker_id] = 1
        
        area = np.sum(mask)
        
        # Filter valid sizes
        if area < min_area or area > max_area:
            continue
        
        masks.append(mask)
    
    # If watershed doesn't work well, fall back to simple contour detection
    if len(masks) < 2:
        print(f"  📍 Watershed detected <2 objects, using contour detection...")
        
        # Find contours directly
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        masks = []
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv.drawContours(mask, [contour], 0, 1, -1)
            masks.append(mask)
    
    if len(masks) == 0:
        print(f"  ⚠️  No significant objects detected")
        return None
    
    print(f"  ✓ Extracted {len(masks)} individual feather masks")
    return masks


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
                 output_dir=None,
                 sam_weights_path="sam3.pt",
                 save_all_masks=False,
                 close_kernel=5,
                 min_component_area=500,
                 stats_csv_path=None,
                 thresholds=None,
                 verbose=True):
    """
    Improved segmentation pipeline that computes mask/bbox/image ratios and optionally filters segments.

    thresholds: dict with optional keys:
        - min_mask_to_bbox: e.g. 0.15    # mask must occupy >= 15% of bbox
        - max_black_ratio:    0.6       # black_in_bbox / bbox_area must be <= 0.6
        - min_mask_to_image:  0.0005    # mask must be >= 0.05% of full image
    """
    if output_dir is None:
        output_dir = _subdir("segmentation")
    if stats_csv_path is None:
        stats_csv_path = os.path.join(_subdir("segmentation"), "segmentation_stats.csv")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(stats_csv_path), exist_ok=True)

    if thresholds is None:
        thresholds = {}

    # Hugging Face (no authentication required - uses local models)
    print("  ℹ️  Using Hugging Face Mobile SAM (no authentication needed)")
    print("  💡 Tip: Set CUDA_VISIBLE_DEVICES=0 to use GPU acceleration")

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

        # Call Replicate API for SAM segmentation
        try:
            results = call_sam3_replicate(image_path, None)
            if results is None or len(results) == 0:
                if verbose:
                    print(f"⚠️  No masks detected in {image_path}")
                continue
        except Exception as e:
            if verbose:
                print(f"❌ Error segmenting {image_path}: {e}")
            continue

        # Process each mask returned by SAM API
        for mask_i, mask_data in enumerate(results):
            mask = mask_data  # mask is already a numpy uint8 array from call_sam3_replicate
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


    return saved, stats_csv_path

def denoise():

    for index, image in enumerate(os.scandir(_subdir("segmentation"))):
        img = cv.imread(image.path)
        dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        filename = os.path.join(_subdir("denoise"), f"denoise_{index}.png")
        cv.imwrite(filename, dst)

def enhance_contrast():
    files = sorted(os.scandir(_subdir("denoise")),
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
        filename = os.path.join(_subdir("contrast"), f"contrast_{index}.png")
        cv.imwrite(filename, enhanced)

def sharpening(strength: float = .7):
    files = sorted(os.scandir("preprocessed/contrast"),
                   key=lambda f: int(f.name.split("_")[1].split(".")[0]))

    for index, image in enumerate(files):
        img = cv.imread(image.path)
        kernel = np.array([[0.0, -1.0, 0.0],
                           [-1.0, 5.0, -1.0],
                           [0.0, -1.0, 0.0]], dtype=np.float32)

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
    # sharpening()
    image_padding()

