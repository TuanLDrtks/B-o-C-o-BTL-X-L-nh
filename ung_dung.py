import gradio as gr

from tien_ich import doc_anh_hoac_csv, chuan_hoa_uint8, luu_png, luu_csv
from bo_loc import (
    loc_trung_binh,
    loc_gauss,
    loc_median,
    nhan_sobel,
    nhan_prewitt,
    dap_ung_laplacian,
    bien_do_gradient,
    nhi_phan_hoa_bien,  # dùng cho ngưỡng nhị phân hoá biên
)


TIEU_DE = "## 🔍 Ứng dụng bộ lọc làm mịn và phát hiện biên"


# =========================================================
# 1. XỬ LÝ LÀM MỊN
# =========================================================
def xu_ly_lam_min(
    tap_tin,
    loai_loc: str,
    kich_thuoc_kernel_lam_min: int,
    sigma_gauss: float,
    kich_thuoc_kernel_median: int,
    kieu_padding: str,
):
    if tap_tin is None:
        raise gr.Error("Vui lòng chọn ảnh đầu vào.")

    try:
        anh_goc = doc_anh_hoac_csv(tap_tin)  # float32, ảnh xám 0–255
    except Exception as e:
        raise gr.Error(str(e))

    anh_goc = anh_goc.astype(float)

    # Chỉ chạy đúng 1 bộ lọc được chọn
    if loai_loc == "Trung bình (Mean)":
        anh_sau = loc_trung_binh(anh_goc, kich_thuoc_kernel_lam_min, kieu_padding)
    elif loai_loc == "Gaussian":
        anh_sau = loc_gauss(
            anh_goc,
            kich_thuoc_kernel_lam_min,
            sigma_gauss,
            kieu_padding,
        )
    elif loai_loc == "Median":
        anh_sau = loc_median(anh_goc, kich_thuoc_kernel_median, kieu_padding)
    else:
        raise gr.Error("Loại bộ lọc làm mịn không hợp lệ.")

    sau_png = luu_png(anh_sau)
    sau_csv = luu_csv(anh_sau)

    return (
        chuan_hoa_uint8(anh_goc),
        chuan_hoa_uint8(anh_sau),
        sau_png,
        sau_csv,
    )


# Ẩn/hiện slider theo loại lọc làm mịn
def cap_nhat_tham_so_lam_min(loai_loc: str):
    if loai_loc == "Trung bình (Mean)":
        return (
            gr.update(visible=True, label="Kích thước kernel Mean (lẻ)"),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif loai_loc == "Gaussian":
        return (
            gr.update(visible=True, label="Kích thước kernel Gaussian (lẻ)"),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    elif loai_loc == "Median":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )


# =========================================================
# 2. XỬ LÝ PHÁT HIỆN BIÊN
# =========================================================
def xu_ly_bien(
    tap_tin,
    loai_bien: str,
    kich_thuoc_kernel_gauss: int,
    sigma_gauss: float,
    kieu_padding: str,
    dung_gauss_truoc_bien: bool,
    nguong_bien: int,  # ngưỡng nhị phân hoá
):
    if tap_tin is None:
        raise gr.Error("Vui lòng chọn ảnh đầu vào.")

    try:
        anh_goc = doc_anh_hoac_csv(tap_tin)
    except Exception as e:
        raise gr.Error(str(e))

    # Tuỳ chọn: làm mịn Gaussian trước khi phát hiện biên
    if dung_gauss_truoc_bien:
        anh_vao = loc_gauss(
            anh_goc,
            kich_thuoc_kernel_gauss,
            sigma_gauss,
            kieu_padding,
        )
    else:
        anh_vao = anh_goc

    # 1) Ảnh biên mức xám (gradient / Laplacian)
    if loai_bien == "Sobel":
        gx, gy = nhan_sobel()
        anh_bien = bien_do_gradient(
            anh_vao,
            gx,
            gy,
            kieu_padding,
            chuan_hoa_0_255=True,
        )
    elif loai_bien == "Prewitt":
        gx, gy = nhan_prewitt()
        anh_bien = bien_do_gradient(
            anh_vao,
            gx,
            gy,
            kieu_padding,
            chuan_hoa_0_255=True,
        )
    elif loai_bien == "Laplacian":
        anh_bien = dap_ung_laplacian(
            anh_vao,
            kieu_padding,
            chuan_hoa_0_255=True,
        )
    else:
        raise gr.Error("Loại bộ lọc biên không hợp lệ.")

    # 2) Nhị phân hoá ảnh biên theo NGƯỠNG
    anh_bien_nhi_phan = nhi_phan_hoa_bien(anh_bien, nguong_bien)

    # 3) Lưu file ảnh biên nhị phân
    bien_png = luu_png(anh_bien_nhi_phan)
    bien_csv = luu_csv(anh_bien_nhi_phan)

    # Trả về:
    #  - Ảnh gốc (xám)
    #  - Ảnh biên nhị phân (0/255)
    #  - File PNG + CSV của ảnh biên nhị phân
    return (
        chuan_hoa_uint8(anh_goc),
        chuan_hoa_uint8(anh_bien_nhi_phan),
        bien_png,
        bien_csv,
    )


# Ẩn/hiện tham số Gaussian trước biên
def cap_nhat_gauss_truoc_bien(dung_gauss: bool):
    vis = True if dung_gauss else False
    return (
        gr.update(visible=vis),
        gr.update(visible=vis),
    )


# =========================================================
# 3. GIAO DIỆN
# =========================================================
def tao_giao_dien() -> gr.Blocks:
    with gr.Blocks(title="Bộ lọc ảnh – Làm mịn & Phát hiện biên") as demo:
        gr.Markdown(TIEU_DE)

        with gr.Tabs():
            # ---------------- TAB 1: LÀM MỊN ẢNH ----------------
            with gr.Tab("✨ Làm mịn ảnh"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1. Ảnh đầu vào")
                        tap_tin_lam_min = gr.File(
                            label="Chọn ảnh PNG/JPG hoặc CSV (ma trận xám)",
                            file_types=["image", ".csv"],
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 2. Chọn bộ lọc & tham số")

                        loai_loc_lam_min = gr.Radio(
                            choices=["Trung bình (Mean)", "Gaussian", "Median"],
                            value="Trung bình (Mean)",
                            label="Chọn 1 bộ lọc làm mịn",
                        )
                        kich_thuoc_kernel_lam_min = gr.Slider(
                            3,
                            15,
                            value=3,
                            step=2,
                            label="Kích thước kernel Mean (lẻ)",
                            visible=True,
                        )
                        sigma_gauss = gr.Slider(
                            0.5,
                            5.0,
                            value=1.0,
                            step=0.1,
                            label="Sigma Gaussian",
                            visible=False,  # mặc định Mean → ẩn
                        )
                        kich_thuoc_kernel_median = gr.Slider(
                            3,
                            15,
                            value=3,
                            step=2,
                            label="Kích thước kernel Median (lẻ)",
                            visible=False,
                        )
                        kieu_padding_lam_min = gr.Radio(
                            choices=["reflect", "replicate", "zero"],
                            value="reflect",
                            label="Kiểu padding biên",
                        )

                nut_lam_min = gr.Button("▶ Chạy lọc làm mịn")

                gr.Markdown("#### 3. So sánh ảnh gốc và ảnh sau làm mịn")

                with gr.Row():
                    anh_goc_lam_min_out = gr.Image(
                        label="Ảnh gốc (xám)", image_mode="L"
                    )
                    anh_sau_lam_min_out = gr.Image(
                        label="Ảnh sau lọc làm mịn", image_mode="L"
                    )

                gr.Markdown("#### 4. Tải kết quả (chỉ ảnh sau lọc)")

                with gr.Row():
                    sau_lam_min_png = gr.File(label="Ảnh sau lọc PNG")
                    sau_lam_min_csv = gr.File(label="Ảnh sau lọc CSV")

                # đổi loại lọc → ẩn/hiện slider tương ứng
                loai_loc_lam_min.change(
                    fn=cap_nhat_tham_so_lam_min,
                    inputs=loai_loc_lam_min,
                    outputs=[
                        kich_thuoc_kernel_lam_min,
                        sigma_gauss,
                        kich_thuoc_kernel_median,
                    ],
                )

                nut_lam_min.click(
                    fn=xu_ly_lam_min,
                    inputs=[
                        tap_tin_lam_min,
                        loai_loc_lam_min,
                        kich_thuoc_kernel_lam_min,
                        sigma_gauss,
                        kich_thuoc_kernel_median,
                        kieu_padding_lam_min,
                    ],
                    outputs=[
                        anh_goc_lam_min_out,
                        anh_sau_lam_min_out,
                        sau_lam_min_png,
                        sau_lam_min_csv,
                    ],
                )

            # ---------------- TAB 2: PHÁT HIỆN BIÊN ----------------
            with gr.Tab("🧪 Phát hiện biên"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1. Ảnh đầu vào")
                        tap_tin_bien = gr.File(
                            label="Chọn ảnh PNG/JPG hoặc CSV (ma trận xám)",
                            file_types=["image", ".csv"],
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 2. Chọn bộ lọc biên & tham số")

                        loai_bien = gr.Radio(
                            choices=["Sobel", "Prewitt", "Laplacian"],
                            value="Sobel",
                            label="Chọn 1 bộ lọc biên",
                        )
                        dung_gauss_truoc_bien = gr.Checkbox(
                            value=True,
                            label="Làm mịn Gaussian trước khi phát hiện biên",
                        )
                        kich_thuoc_kernel_gauss = gr.Slider(
                            3,
                            15,
                            value=3,
                            step=2,
                            label="Kích thước kernel Gaussian (lẻ)",
                            visible=True,
                        )
                        sigma_gauss_bien = gr.Slider(
                            0.5,
                            5.0,
                            value=1.0,
                            step=0.1,
                            label="Sigma Gaussian",
                            visible=True,
                        )
                        kieu_padding_bien = gr.Radio(
                            choices=["reflect", "replicate", "zero"],
                            value="reflect",
                            label="Kiểu padding biên",
                        )
                        # Slider NGƯỠNG BIÊN
                        nguong_bien = gr.Slider(
                            minimum=0,
                            maximum=255,
                            value=100,
                            step=1,
                            label="Ngưỡng nhị phân hoá biên",
                        )

                nut_bien = gr.Button("▶ Chạy phát hiện biên")

                gr.Markdown("#### 3. So sánh ảnh gốc và ảnh biên")

                with gr.Row():
                    anh_goc_bien_out = gr.Image(
                        label="Ảnh gốc (xám)", image_mode="L"
                    )
                    anh_bien_out = gr.Image(
                        label="Ảnh biên nhị phân (0/255)",
                        image_mode="L",
                    )

                gr.Markdown("#### 4. Tải kết quả (chỉ ảnh biên)")

                with gr.Row():
                    bien_png = gr.File(label="Ảnh biên PNG")
                    bien_csv = gr.File(label="Ảnh biên CSV")

                # Ẩn/hiện tham số Gaussian trước biên
                dung_gauss_truoc_bien.change(
                    fn=cap_nhat_gauss_truoc_bien,
                    inputs=dung_gauss_truoc_bien,
                    outputs=[kich_thuoc_kernel_gauss, sigma_gauss_bien],
                )

                nut_bien.click(
                    fn=xu_ly_bien,
                    inputs=[
                        tap_tin_bien,
                        loai_bien,
                        kich_thuoc_kernel_gauss,
                        sigma_gauss_bien,
                        kieu_padding_bien,
                        dung_gauss_truoc_bien,
                        nguong_bien,
                    ],
                    outputs=[
                        anh_goc_bien_out,
                        anh_bien_out,
                        bien_png,
                        bien_csv,
                    ],
                )

        return demo


if __name__ == "__main__":
    app = tao_giao_dien()
    app.launch()
