# bo_loc/bien.py
from typing import Literal, Tuple

import numpy as np

from .cong_cu_chap import chap_2d, KieuPadding
from tien_ich import chuan_hoa_uint8

KieuPadding = Literal["zero", "replicate", "reflect"]


def nhan_sobel() -> Tuple[np.ndarray, np.ndarray]:
    """Trả về (Gx, Gy) cho Sobel."""
    gx = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=float,
    )
    gy = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=float,
    )
    return gx, gy


def nhan_prewitt() -> Tuple[np.ndarray, np.ndarray]:
    """Trả về (Gx, Gy) cho Prewitt."""
    gx = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ],
        dtype=float,
    )
    gy = np.array(
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=float,
    )
    return gx, gy


def nhan_laplacian() -> np.ndarray:
    """Kernel Laplacian 4-neighborhood."""
    return np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ],
        dtype=float,
    )


def bien_do_gradient(
    anh_xam: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    kieu_padding: KieuPadding,
    chuan_hoa_0_255: bool = True,
) -> np.ndarray:
    """Độ lớn gradient dùng 2 kernel gx, gy (Sobel/Prewitt)."""
    fx = chap_2d(anh_xam, gx, kieu_padding)
    fy = chap_2d(anh_xam, gy, kieu_padding)
    mag = np.sqrt(fx**2 + fy**2)

    if chuan_hoa_0_255:
        mn, mx = float(mag.min()), float(mag.max())
        if mx > mn:
            mag = (mag - mn) * 255.0 / (mx - mn)

    return mag


def dap_ung_laplacian(
    anh_xam: np.ndarray,
    kieu_padding: KieuPadding,
    chuan_hoa_0_255: bool = True,
) -> np.ndarray:
    """Độ lớn đáp ứng Laplacian (lấy trị tuyệt đối)."""
    k = nhan_laplacian()
    resp = chap_2d(anh_xam, k, kieu_padding)
    resp = np.abs(resp)

    if chuan_hoa_0_255:
        mn, mx = float(resp.min()), float(resp.max())
        if mx > mn:
            resp = (resp - mn) * 255.0 / (mx - mn)

    return resp


def nhi_phan_hoa_bien(anh_bien: np.ndarray, nguong: int) -> np.ndarray:
    """Nhị phân hoá ảnh biên (0/255) theo ngưỡng."""
    return (anh_bien >= float(nguong)).astype(np.uint8) * 255


def ve_bien_len_anh_xam(anh_xam: np.ndarray, bien_nhi_phan: np.ndarray) -> np.ndarray:
    """
    Overlay biên (bien_nhi_phan > 0) màu đỏ lên ảnh xám.
    Trả về ảnh RGB uint8.
    """
    g = chuan_hoa_uint8(anh_xam)
    e = chuan_hoa_uint8(bien_nhi_phan)

    H, W = g.shape
    rgb = np.stack([g, g, g], axis=-1)
    mask = e > 0

    rgb[mask, 0] = 255  # R
    rgb[mask, 1] = 0
    rgb[mask, 2] = 0

    return rgb.astype(np.uint8)
