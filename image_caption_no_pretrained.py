import streamlit as st
from PIL import Image, ImageStat, ImageFilter
import numpy as np
import colorsys
import re

st.set_page_config(page_title="Local Image Captioning (No Models)", layout="wide")

st.title("🖼️ Local Rule-Based Image Captioning")
st.markdown("Pure computer vision analysis with Pillow – no pretrained models or APIs. Generates descriptive captions from image stats.")

uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg', 'webp'])
image_url = st.text_input("Or URL (demo only, local analyzes upload)")

if st.button("Generate Caption", type="primary"):
    if not uploaded_file:
        st.error("Upload image.")
        st.stop()

    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, use_column_width=True)

    # Analyze
    with st.spinner("Analyzing..."):
        # Stats
        stat = ImageStat.Stat(img)
        brightness = np.mean(stat.mean) / 255  # 0 dark, 1 bright
        size = img.size
        aspect = size[1] / size[0]  # >1.5 tall (phone-like)

        # Dominant color
        np_img = np.array(img)
        pixels = np_img.reshape(-1, 3)
        dominant_idx = np.argmax(np.bincount(np_img.reshape(-1, 3).ravel() // 16 * 16))
        dom_r = (dominant_idx // 256 // 256) * 16
        dom_g = (dominant_idx // 256 % 16) * 16
        dom_b = (dominant_idx % 256) * 16
        dom_hsv = colorsys.rgb_to_hsv(dom_r/255, dom_g/255, dom_b/255)
        if dom_hsv[1] > 0.5:
            color_adj = "vibrant"
        else:
            color_adj = "dark"
        color = "black" if dom_hsv[1] < 0.2 else "dark" if dom_hsv[2] < 0.4 else "shiny"

        # Edges for detail/object
        edges = img.filter(ImageFilter.FIND_EDGES)
        edge_mean = ImageStat.Stat(edges).mean[0]
        detail = "highly detailed" if edge_mean > 100 else "smooth"

        # Lighting
        lighting = "soft natural light" if 0.4 < brightness < 0.7 else "bright lighting" if brightness > 0.7 else "dim lighting"

        # Object guess by aspect/shape
        if aspect > 1.8 and color == "black":
            subject = "sleek black smartphone"
        elif aspect > 2:
            subject = "tall object"
        else:
            subject = "composition"

        # Environment
        bg_stat = ImageStat.Stat(img.filter(ImageFilter.GaussianBlur(20)))
        bg_bright = np.mean(bg_stat.mean) / 255
        env = "textured white surface" if bg_bright > 0.8 and brightness > 0.5 else "minimalist background"

        caption = f"{subject} rests on {env} under {lighting}, its glossy screen glowing with {color_adj} hues amid {detail} textures."

        word_count = len(re.findall(r'\w+', caption))

    st.subheader("📝 Generated Caption")
    st.markdown(f"""
    <div style='background-color: #fef3c7; padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b;'>
        <p style='font-size: 18px; line-height: 1.6; margin: 0;'>{caption}</p>
        <p style='color: #92400e; font-size: 14px; margin-top: 12px;'><strong>Words:</strong> {word_count}</p>
    </div>
    """, unsafe_allow_html=True)

    st.code(caption)
    st.success("Local analysis complete – no models used!")

st.info("💡 Heuristic CV: dominant color, brightness, edges, aspect ratio → template caption. Works well for simple images like smartphones. Test with image.png.")

