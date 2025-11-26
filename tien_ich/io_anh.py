# tien_ich/io_anh.py
import os
import tempfile
from typing import Any

import numpy as np
from PIL import Image


def doc_anh_hoac_csv(tap_tin: Any) -> np.ndarray:
    """
    Đọc ảnh từ file (Gradio File):
    - Ảnh PNG/JPG/BMP/TIF → chuyển sang ảnh xám, nếu quá lớn thì thu nhỏ lại.
    - CSV → đọc ma trận, nếu max <= 1 thì scale lên 0–255.

    Trả về: ndarray float32 (H, W) với giá trị 0–255.
    """
    if tap_tin is None:
        raise ValueError("Chưa chọn file đầu vào.")

    ten = getattr(tap_tin, "name", "")
    _, phu = os.path.splitext(ten.lower())

    # Ảnh thật
    if phu in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        img = Image.open(tap_tin).convert("L")  # chuyển sang xám

        # Giới hạn kích thước để xử lý nhanh
        canh_max = max(img.size)
        if canh_max > 1024:
            ti_le = 1024.0 / canh_max
            kich_thuoc_moi = (int(img.size[0] * ti_le), int(img.size[1] * ti_le))
            img = img.resize(kich_thuoc_moi, Image.BILINEAR)

        arr = np.array(img, dtype=np.float32)
        return arr

    # CSV
    if hasattr(tap_tin, "seek"):
        tap_tin.seek(0)

    du_lieu = np.loadtxt(tap_tin, delimiter=",")
    if du_lieu.ndim != 2:
        raise ValueError("CSV phải là ma trận 2 chiều.")

    du_lieu = du_lieu.astype(np.float32)
    mmax = float(du_lieu.max())
    if mmax <= 1.0:
        du_lieu = du_lieu * 255.0

    du_lieu = np.clip(du_lieu, 0, 255)
    return du_lieu


def chuan_hoa_uint8(anh: np.ndarray) -> np.ndarray:
    """Đưa ảnh về uint8 (0–255), xử lý NaN/Inf nếu có."""
    anh = np.nan_to_num(anh)
    anh = np.clip(anh, 0, 255)
    return anh.astype(np.uint8)


def luu_png(anh: np.ndarray) -> str:
    """Lưu ảnh ra file PNG tạm để tải về."""
    anh_u8 = chuan_hoa_uint8(anh)
    pil = Image.fromarray(anh_u8)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pil.save(tmp, format="PNG")
    tmp.close()
    return tmp.name


def luu_csv(anh: np.ndarray) -> str:
    """Lưu ma trận ảnh ra file CSV, trả về đường dẫn file."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    np.savetxt(tmp, anh, fmt="%.4f", delimiter=",")
    tmp.close()
    return tmp.name
