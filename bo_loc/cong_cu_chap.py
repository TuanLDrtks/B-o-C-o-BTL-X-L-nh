# bo_loc/cong_cu_chap.py
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import as_strided

KieuPadding = Literal["zero", "replicate", "reflect"]


def them_le(anh: np.ndarray, k: int, kieu: KieuPadding) -> np.ndarray:
    """
    Thêm lề (padding) cho ảnh.
    k: bán kính kernel (kernel_size = 2k + 1)
    """
    if kieu == "zero":
        return np.pad(anh, ((k, k), (k, k)), mode="constant", constant_values=0)
    if kieu == "replicate":
        return np.pad(anh, ((k, k), (k, k)), mode="edge")
    if kieu == "reflect":
        return np.pad(anh, ((k, k), (k, k)), mode="reflect")
    raise ValueError("Kiểu padding không hợp lệ.")


def chap_2d(anh: np.ndarray, nhan: np.ndarray, kieu_padding: KieuPadding) -> np.ndarray:
    """
    Chập 2D (thực chất là tương quan 2D) giữa ảnh xám và kernel.

    Cài bằng numpy (as_strided + einsum) nên nhanh hơn vòng for thuần.
    nhan: kernel vuông, kích thước lẻ (3x3, 5x5, ...).
    """
    ks = nhan.shape[0]
    assert ks == nhan.shape[1] and ks % 2 == 1, "Kernel phải vuông và lẻ."

    k = ks // 2
    anh_mo_rong = them_le(anh, k, kieu_padding)
    anh_mo_rong = np.ascontiguousarray(anh_mo_rong)

    H, W = anh.shape
    s0, s1 = anh_mo_rong.strides

    # Tạo "cửa sổ trượt" (H, W, ks, ks) trên ảnh mở rộng
    cua_so = as_strided(
        anh_mo_rong,
        shape=(H, W, ks, ks),
        strides=(s0, s1, s0, s1),
    )

    # Tính tổng nhân từng cửa sổ với kernel
    ket_qua = np.einsum("ij,xyij->xy", nhan, cua_so, optimize=True)
    return ket_qua
