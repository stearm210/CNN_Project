import streamlit as st
import torch
from model import SRResNet, SRResNet_MLKA
from utils import convert_image
from PIL import Image

# 设置页面标题
st.set_page_config(page_title="Reconstruction", page_icon="🚀", layout="wide")


# 创建容器
col1, col2, col3 = st.columns(3)

with col2:
    st.title("Reconstruction")
    display_mode = st.radio(
        "Display mode",
        ("Single screen display", "Dual screen display")
    )
    start_button = st.button("Start running")

# 创建侧边栏
sidebar = st.sidebar

# 添加大标题：参数设置
sidebar.header("Parameter settings")

# 添加下拉选择栏
model_name = sidebar.selectbox(
    "Select Model",
    ("SRResNet", "SRResNet(MLKA)")
)

# 添加大标题：识别文件类型设置
sidebar.header("Identify file type settings")

# 添加文件类型选择下拉栏
file_type = sidebar.selectbox(
    "File type",
    ("Image")
)

# 根据文件类型选择下拉栏中选择的文件类型，动态显示不同的文件上传功能
if file_type == "Image":
    uploaded_file = sidebar.file_uploader("upload image", type=["png", "jpg", "jpeg"])
# elif file_type == "Video":
#     uploaded_file = sidebar.file_uploader("upload video", type=["mp4", "avi", "mov", "wmv"])

def process_image(model_name, uploaded_file):
    try:
        # 直接从上传的文件对象加载图像
        img = Image.open(uploaded_file).convert('RGB')

        large_kernel_size = 9
        small_kernel_size = 3
        n_blocks = 16
        scale = 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "SRResNet":
            premodel = "results/checkpoint_srresnet1.pth"
            n_channels = 64
            checkpoint = torch.load(premodel)
            srresnet = SRResNet(large_kernel_size=large_kernel_size,
                                small_kernel_size=small_kernel_size,
                                n_channels=n_channels,
                                n_blocks=n_blocks,
                                scaling_factor=scale).to(device)
            srresnet.load_state_dict(checkpoint['model'])
            srresnet.eval()

            lr_img = convert_image(img, source='pil', target='imagenet-norm')
            lr_img.unsqueeze_(0)
            lr_img = lr_img.to(device)

            with torch.no_grad():
                sr_img = srresnet(lr_img).squeeze(0).cpu().detach()
                sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')

            return sr_img

        else:
            premodel = "results/SRResNet_MLKA1.pth"
            n_channels = 72
            checkpoint = torch.load(premodel)
            srresnet = SRResNet_MLKA(large_kernel_size=large_kernel_size,
                                small_kernel_size=small_kernel_size,
                                n_channels=n_channels,
                                n_blocks=n_blocks,
                                scaling_factor=scale).to(device)
            srresnet.load_state_dict(checkpoint['model'])
            srresnet.eval()

            lr_img = convert_image(img, source='pil', target='imagenet-norm')
            lr_img.unsqueeze_(0)
            lr_img = lr_img.to(device)

            with torch.no_grad():
                sr_img = srresnet(lr_img).squeeze(0).cpu().detach()
                sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')

            return sr_img

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
        return None

default_image = Image.open("default_image.jpg").resize((600, 600))

# 展示处理后的图像
if display_mode == "Single screen display":
    if uploaded_file is not None and start_button:
        processed_image = process_image(model_name, uploaded_file)
        col1, col2, col3 = st.columns(3)
        if processed_image is not None:
            with col2:
                st.image(processed_image.resize((600, 600)), caption='Reconstruct image')
        else:
            col1, col2, col3 = st.columns(3)
            with col2:
                st.image(default_image, caption='Default Image')
    else:
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image(default_image, caption='Default Image')
elif display_mode == "Dual screen display":
    if uploaded_file is not None and start_button:
        processed_image = process_image(model_name, uploaded_file)
        if processed_image is not None:
            # 使用两个列展示原始和重建图片
            col1, col2 = st.columns(2)
            with col1:
                original_image = Image.open(uploaded_file).resize((600, 600))
                st.image(original_image, caption='Original image')
            with col2:
                st.image(processed_image.resize((600, 600)), caption='Reconstruct image')
        else:
            st.image(default_image, caption='Default Image')
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(default_image, caption='Default Image')
        with col2:
            st.image(default_image, caption="Default Image")