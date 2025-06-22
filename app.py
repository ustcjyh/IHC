import os, io, zipfile, numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from skimage import io as skio, measure, segmentation
from skimage.color import separate_stains, hed_from_rgb
from cellpose import models

st.set_page_config(page_title="IHC Cellpose App", layout="wide")
st.title("🧬 IHC Cellpose App (v3 compatible)")

with st.sidebar:
    dab_thr   = st.slider("DAB 阳性阈值", 0.0, 1.0, 0.10, 0.01)
    flow_thr  = st.slider("flow_threshold", -0.4, 1.0, 0.4, 0.05)
    min_size  = st.slider("最小细胞面积", 0, 200, 20, 5)

@st.cache_resource
def load_model():
    try:
        return models.CellposeModel(gpu=False, model_type='nuclei')
    except AttributeError:
        return models.Cellpose(model_type='nuclei')
model = load_model()

uploaded = st.file_uploader("上传图像或 ZIP 文件", type=['png','tif','jpg','zip'])
if uploaded:
    files = []
    if uploaded.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded) as zf:
            for name in zf.namelist():
                if name.lower().endswith(('.png','.tif','.jpg')):
                    files.append( (name, io.BytesIO(zf.read(name))) )
    else:
        files.append( (uploaded.name, uploaded) )

    all_cells, image_stats = [], []
    for fname, file in files:
        img = skio.imread(file)[:, :, :3]
        dab = separate_stains(img, hed_from_rgb)[:, :, 2]
        result = model.eval(img, channels=[0, 0], flow_threshold=flow_thr,
                            min_size=min_size, diameter=None)
        masks = result['masks']
        pos_mask = np.zeros_like(masks)
        cells = []
        pos_cnt = 0
        for r in measure.regionprops(masks, intensity_image=dab):
            status = 'positive' if r.mean_intensity > dab_thr else 'negative'
            if status == 'positive':
                pos_mask[masks == r.label] = 1
                pos_cnt += 1
            cells.append([fname, r.label, r.area, r.mean_intensity, status])
        all_cells.extend(cells)
        image_stats.append([fname, len(cells), pos_cnt, pos_cnt/len(cells)*100 if cells else 0])
        st.image(segmentation.mark_boundaries(img, pos_mask, color=(1,0,0)),
                 caption=f"{fname} - Positive: {pos_cnt}/{len(cells)}")

    df1 = pd.DataFrame(all_cells, columns=["Image","Cell ID","Area","Mean DAB","Status"])
    df2 = pd.DataFrame(image_stats, columns=["Image","Total","Positive","Positive %"])
    st.dataframe(df2)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("ihc_cellwise_results.csv", df1.to_csv(index=False))
        zf.writestr("ihc_image_summary.csv", df2.to_csv(index=False))
    st.download_button("⬇️ 下载分析结果 ZIP", zip_buffer.getvalue(), "IHC_results.zip", "application/zip")
