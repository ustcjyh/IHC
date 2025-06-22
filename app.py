import os, io, zipfile, numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from skimage import io as skio, measure, segmentation, transform
from skimage.color import separate_stains, hed_from_rgb
from cellpose import models

st.set_page_config(page_title="IHC Cellpose App (Lite)", layout="wide")
st.title("🧬 IHC Cellpose App (Lite / Cellpose v3.1)")

with st.sidebar:
    dab_thr   = st.slider("DAB 阈值 (阳性判定)", 0.0, 1.0, 0.10, 0.01)
    flow_thr  = st.slider("flow_threshold", -0.4, 1.0, 0.4, 0.05)
    min_size  = st.slider("最小细胞面积", 0, 200, 20, 5)
    max_files = st.slider("最多分析图像数", 1, 10, 3, 1)

@st.cache_resource
def load_model():
    return models.Cellpose(model_type='nuclei')

model = load_model()

uploaded = st.file_uploader("上传图像或 ZIP（建议 <3 张图）", type=['png','tif','jpg','zip'])
if uploaded:
    files = []
    if uploaded.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(('.png','.jpg','.tif'))]
            if len(names) > max_files:
                st.warning(f"⚠️ ZIP 图像数量过多，仅处理前 {max_files} 张。")
            for name in names[:max_files]:
                files.append((name, io.BytesIO(zf.read(name))))
    else:
        files.append((uploaded.name, uploaded))

    all_cells, image_stats = [], []
    for fname, file in files:
        img = skio.imread(file)[:, :, :3]
        if max(img.shape) > 1024:
            scale = 1024 / max(img.shape)
            img = transform.rescale(img, scale, channel_axis=2, anti_aliasing=True)
            img = (img * 255).astype(np.uint8)

        dab = separate_stains(img, hed_from_rgb)[:, :, 2]
        masks, _, _, _ = model.eval(img, channels=[0, 0], flow_threshold=flow_thr,
                                    min_size=min_size, diameter=None)

        pos_mask = np.zeros_like(masks)
        cells, pos_cnt = [], 0
        for r in measure.regionprops(masks, intensity_image=dab):
            status = 'positive' if r.mean_intensity > dab_thr else 'negative'
            if status == 'positive':
                pos_mask[masks == r.label] = 1
                pos_cnt += 1
            cells.append([fname, r.label, r.area, r.mean_intensity, status])
        all_cells.extend(cells)
        image_stats.append([fname, len(cells), pos_cnt, pos_cnt / len(cells) * 100 if cells else 0])
        st.image(segmentation.mark_boundaries(img, pos_mask, color=(1, 0, 0)),
                 caption=f"{fname}: Positive {pos_cnt}/{len(cells)}")

    df1 = pd.DataFrame(all_cells, columns=["Image", "Cell ID", "Area", "Mean DAB", "Status"])
    df2 = pd.DataFrame(image_stats, columns=["Image", "Total", "Positive", "Positive %"])
    st.dataframe(df2)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("ihc_cellwise_results.csv", df1.to_csv(index=False))
        zf.writestr("ihc_image_summary.csv", df2.to_csv(index=False))
    st.download_button("⬇️ 下载分析结果", zip_buffer.getvalue(), "IHC_results.zip", "application/zip")
