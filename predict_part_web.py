import sys
import warnings
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from detectron2.utils.visualizer import Visualizer
from predict_part import get_predictor, InstancesWrap


@st.cache_resource # avoid reloading
def _get_predcitor(cfg, ckpt):
    print(f"loading model from {ckpt}")
    return get_predictor(cfg, ckpt)

@st.cache_data # avoid reloading
def get_image(st_image):
    # remove alpha channel if exists
    image = Image.open(st_image)
    if image.mode == "RGBA":
        image = Image.fromarray(np.asarray(image)[:, :, :3])
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    return image, w, h

def run():
    st.title("Open World Part Segmentation")
    predictor = _get_predcitor(
        "configs/part_segmentation/dt_alpha/clsag.yaml", 
        sys.argv[1],
    )
    st_image = st.file_uploader("Image", ["jpg", "png"])
    if st_image:
        image, w, h = get_image(st_image)
        
        fill_color = "rgba(255, 255, 255, 0.0)"
        stroke_width = st.slider("Brush Size for drawing mask", min_value=1, max_value=100, value=10, step=1)
        stroke_color = "rgba(255, 255, 255, 1.0)"
        bg_color = "rgba(0, 0, 0, 1.0)"
        drawing_mode = "freedraw"
        st.write("Canvas")
        st.caption(
            "Draw a mask to for part prediction, then click the 'Send to Streamlit' button (bottom left, with an arrow on it).")
        canvas_result = st_canvas(
            fill_color=fill_color,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=image,
            update_streamlit=False,
            height=h,
            width=w,
            drawing_mode=drawing_mode,
            key="canvas",
        )

        with st.form("form"): # avoid reloading
            threshold = st.slider("score threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            topk = st.number_input("topk", min_value=1, max_value=1000000, value=10)
            submitted = st.form_submit_button("GO!")
            if canvas_result and submitted:
                mask = canvas_result.image_data
                mask = mask[:, :, -1] > 0
                if mask.sum() > 0:
                    input = np.asarray(image)
                    input = np.append(input, np.asarray(mask)[:, :, None].astype(np.uint8) * 255, axis=2)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        outputs = predictor(input)
                    outputs["instances"] = outputs["instances"].to("cpu")
                    instances = InstancesWrap(outputs["instances"], threshold, topk)
                    st.write("Predicted!")
                    v = Visualizer(input[:, :, :3], scale=1.0)
                    out = v.draw_instance_predictions(instances)
                    out = Image.fromarray(out.get_image())
                    st.image(out, output_format='PNG')
                else:
                    st.write("no mask!")
                    st.caption(
                        "remeber to click the 'Send to Streamlit' button (bottom left, with an arrow on it) after drawing a mask.")


if __name__ == "__main__":
    run()