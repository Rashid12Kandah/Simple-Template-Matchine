# 🥦🍌 Sliding-Window Template Matching for Object Detection

A from-scratch implementation of object detection using a **sliding window** and **Mean Squared Error (MSE)** for template matching — no pre-trained models, no high-level matching APIs. Built and tested in **Google Colab**.

The notebook locates a **banana** and a **broccoli** inside a larger grid image of fruits and vegetables, draws a bounding box around the best match, and labels it.

---

## 📌 Overview

The pipeline is intentionally minimal so the matching mechanics are visible end-to-end:

1. **Load** the source image and convert it to grayscale.
2. **Extract templates** for the banana and broccoli by slicing regions out of the source image.
3. **Slide a window** of the same size as each template across the full image.
4. **Score every window** against the template using MSE — lower is better.
5. **Pick the minimum-MSE position** as the detection.
6. **Annotate** the original image with a bounding box and the object label.

---

## 🧪 Built for Google Colab

This notebook was developed for **Google Colab**, and that decision shapes one important detail of the code.

`cv2.imshow()` **does not work in Colab**. Colab runs as a hosted Jupyter environment with no GUI backend, so calling `cv2.imshow()` will crash the runtime. Google ships a drop-in replacement in `google.colab.patches`:

```python
from google.colab.patches import cv2_imshow

cv2_imshow(image)   # ✅ works in Colab
# cv2.imshow("win", image)  # ❌ crashes the Colab kernel
```

> **Running locally instead?** Replace every `cv2_imshow(img)` with:
> ```python
> cv2.imshow("window", img)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```

---

## 🛠️ Tech Stack

| Library | Role |
|---|---|
| `cv2` (OpenCV) | Image I/O, grayscale conversion, drawing, text |
| `numpy` | Array slicing for template extraction |
| `skimage.metrics.mean_squared_error` | Similarity metric between window and template |
| `google.colab.patches.cv2_imshow` | Colab-safe image display |

---

## 🚀 Quickstart

### Open in Colab
1. Upload `Simple-Template-Matching.ipynb` to your Google Drive, or open it directly in Colab.
2. Upload `veget.jpg` to the Colab runtime (left sidebar → Files → Upload).
3. Run the cells top-to-bottom.

### Run locally
```bash
pip install opencv-python numpy scikit-image
jupyter notebook Simple-Template-Matching.ipynb
```
Remember to swap `cv2_imshow` calls for the local equivalent shown above.

---

## 🧩 How It Works

### 1. Load and grayscale the source image
Working in grayscale keeps the MSE comparison single-channel and fast.

```python
img = cv2.imread('veget.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Image Shape:", img.shape)   # (1680, 1920)
```

### 2. Extract templates by slicing
Templates are cropped directly from the source image using NumPy slicing:

```python
banana   = img[0:420,    1500:1920]   # top-right cell
brocoli  = img[421:840,  481:900]     # second row, second cell
```

### 3. Sliding window generator
A simple generator yields every window position at a given step size:

```python
def sliding_window(image, step, windowSize):
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
```

### 4. Score with MSE
Lower MSE = the window looks more like the template.

```python
from skimage.metrics import mean_squared_error as mse

def compare_images(imageA, imageB):
    return mse(imageA, imageB)
```

### 5. Find the minimum and annotate
Each `(error, x, y)` tuple is collected, sorted, and the top result is drawn onto the image with `cv2.rectangle` and `cv2.putText`.

---

## ⚙️ Parameters

| Parameter | Value | Notes |
|---|---|---|
| `step` | `50` px | Smaller = more accurate but quadratically slower |
| Template size — banana | 420 × 420 | Matches a single grid cell |
| Template size — broccoli | 419 × 419 | Matches a single grid cell |
| Metric | MSE | Sensitive to absolute pixel differences |

---

## ⚠️ Limitations

This is intentionally a teaching implementation. It is **not** a production detector.

- **Not scale-invariant** — templates must be the same size as the target.
- **Not rotation-invariant** — a rotated banana would not match.
- **Sensitive to lighting and color shifts** — MSE punishes any pixel-level difference.
- **Slow** — brute-force sliding window scales as `O(W·H / step²)`.
- For real-world use, look at `cv2.matchTemplate`, feature-based methods (SIFT/ORB), or a CNN-based detector.

---

## 📁 Repository Structure

```
.
├── Simple-Template-Matching.ipynb   # Main notebook
├── veget.jpg                                    # Source image (grid of fruits/vegetables)
└── README.md
```

---

## 📚 Context

Built as an academic assignment exploring the fundamentals of template matching before moving on to feature-based and learning-based detectors. The goal was to make every step of the matching loop explicit, rather than calling a single high-level API.

---

## 📝 License

Free to use for educational purposes.
