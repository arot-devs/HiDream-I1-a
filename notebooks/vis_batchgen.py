import os
import json
from datetime import datetime
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import unibox as ub

st.set_page_config(page_title="Image Gallery Viewer", layout="wide")

# @st.cache_data
# def load_gallery_data(root_folders):
    # entries = []
    # for root in root_folders:
    #     files = ub.traverses(root)
    #     # images = [f for f in files if f.endswith('.jpg') and '_annotated' not in f]

    #     # after sorting, remove latest 4 images to avoid io error
    #     images = sorted([f for f in files])[4:]
                        
    #     jsons = [f for f in files if f.endswith('.json')]

    #     for image_path in images:
    #         # Expected format: <timestamp>.jpg and <timestamp>.json
    #         base = os.path.splitext(image_path)[0]
    #         json_path = base + ".json"
    #         if json_path in jsons:
    #             try:
    #                 with open(json_path, 'r') as jf:
    #                     meta = json.load(jf)
    #                 entries.append({
    #                     "image_file": image_path,
    #                     "prompt": meta.get("prompt", ""),
    #                     "model": meta.get("model", ""),
    #                     "time": datetime.fromisoformat(meta.get("date")),
    #                 })
    #             except Exception:
    #                 continue
    # entries.sort(key=lambda x: x['time'], reverse=True)
    # return entries

from PIL import Image, UnidentifiedImageError

@st.cache_data
def load_gallery_data(root_folders):
    entries = []
    invalid_images = []

    for root in root_folders:
        files = ub.ls(root)
        images = sorted([f for f in files if ".png" in f])[4:]
        jsons = [f for f in files if f.endswith('.json')]

        for image_path in images:
            base = os.path.splitext(image_path)[0]
            json_path = base + ".json"
            if json_path in jsons:
                try:
                    # Validate image before loading metadata
                    with open(image_path, 'rb') as img_file:
                        Image.open(img_file).verify()

                    with open(json_path, 'r') as jf:
                        meta = json.load(jf)

                    entries.append({
                        "image_file": image_path,
                        "prompt": meta.get("prompt", ""),
                        "model": meta.get("model", ""),
                        "time": datetime.fromisoformat(meta.get("date")),
                    })
                except (UnidentifiedImageError, OSError):
                    invalid_images.append(image_path)
                except Exception:
                    continue

    entries.sort(key=lambda x: x['time'], reverse=True)

    # Show invalids in Streamlit (if any)
    if invalid_images:
        st.text("Invalid or corrupted image files:")
        for path in invalid_images:
            st.text(f" - {path}")

    return entries

def show_images(data, page, page_size):
    subset = data[page * page_size:(page + 1) * page_size]
    print(subset[0])
    cols = st.columns(4)
    for i, item in enumerate(subset):
        with cols[i % 4]:
            st.image(item['image_file'], use_container_width=True)
            st.caption(item['prompt'])

def main(root_folders):
    st_autorefresh(interval=5 * 60 * 1000)  # Refresh every 5 minutes

    st.title("Image Gallery Viewer")
    st.write(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    folder_labels = ['<all>'] + [os.path.basename(f) for f in root_folders]
    selected = st.sidebar.selectbox("Choose folder", folder_labels)

    selected_folders = root_folders if selected == "<all>" else [
        f for f in root_folders if os.path.basename(f) == selected
    ]

    entries = load_gallery_data(selected_folders)

    filename_filter = st.sidebar.text_input("Filename contains").strip().lower()
    prompt_filter = st.sidebar.text_input("Prompt contains").strip().lower()

    if filename_filter:
        entries = [e for e in entries if filename_filter in os.path.basename(e['image_file']).lower()]
    if prompt_filter:
        entries = [e for e in entries if prompt_filter in e['prompt'].lower()]

    if not entries:
        st.write("No matching entries found.")
        return

    page_size = 100
    total_pages = (len(entries) + page_size - 1) // page_size
    page = st.sidebar.number_input("Page", 0, max(total_pages - 1, 0), 0)

    show_images(entries, page, page_size)

if __name__ == "__main__":
    # pip install streamlit_autorefresh streamlit 
    # streamlit run vis_batchgen.py
    root_folders = [
        "/local/yada/apps/HiDream-I1-a/outputs_user_prompts",
    ]
    main(root_folders)
