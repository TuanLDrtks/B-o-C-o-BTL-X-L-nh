# bo_loc/lam_min.py
from typing import Literal

import numpy as np

from .cong_cu_chap import chap_2d, them_le, KieuPadding

KieuPadding = Literal["zero", "replicate", "reflect"]


def nhan_trung_binh(kich_thuoc: int) -> np.ndarray:
    """Tạo kernel trung bình (mean)."""
    return np.ones((kich_thuoc, kich_thuoc), dtype=float) / float(
        kich_thuoc * kich_thuoc
    )


def nhan_gauss(kich_thuoc: int, sigma: float) -> np.ndarray:
    """Tạo kernel Gaussian 2D, chuẩn hóa tổng = 1."""
    assert kich_thuoc % 2 == 1, "Kích thước kernel Gaussian phải lẻ."

    truc = np.arange(-(kich_thuoc // 2), kich_thuoc // 2 + 1, dtype=float)
    xx, yy = np.meshgrid(truc, truc)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    k /= k.sum()
    return k


def loc_trung_binh(
    anh_xam: np.ndarray, kich_thuoc: int, kieu_padding: KieuPadding
) -> np.ndarray:
    """Lọc trung bình (mean filter)."""
    k = nhan_trung_binh(kich_thuoc)
    return chap_2d(anh_xam, k, kieu_padding)


def loc_gauss(
    anh_xam: np.ndarray, kich_thuoc: int, sigma: float, kieu_padding: KieuPadding
) -> np.ndarray:
    """Lọc Gaussian."""
    k = nhan_gauss(kich_thuoc, sigma)
    return chap_2d(anh_xam, k, kieu_padding)


def loc_median(
    anh_xam: np.ndarray, kich_thuoc: int, kieu_padding: KieuPadding
) -> np.ndarray:
    """
    Lọc median tự cài:
    - padding theo mode
    - lấy median trong cửa sổ kích_thuoc x kích_thuoc
    """
    assert kich_thuoc % 2 == 1, "Kích thước kernel Median phải lẻ."

    ban_kinh = kich_thuoc // 2
    anh_mo_rong = them_le(anh_xam, ban_kinh, kieu_padding)
    H, W = anh_xam.shape
    ket_qua = np.zeros((H, W), dtype=float)

    for i in range(H):
        for j in range(W):
            cua_so = anh_mo_rong[i : i + kich_thuoc, j : j + kich_thuoc]
            ket_qua[i, j] = float(np.median(cua_so))

    return ket_qua
