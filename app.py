import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="SVD Image Compression", page_icon="🧮", layout="wide")
st.title("SVD-Based Image Compression")
st.caption("Main project: grayscale image compression. Extensions: RGB colour compression and an eigenvalue demonstration.")

def load_image(file, max_side=256):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(file.getvalue()))).convert("RGB")
    original_size = image.size
    image.thumbnail((max_side, max_side))
    return np.asarray(image, dtype=np.float64), original_size

def svd_reconstruct(matrix, k):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rebuilt = (u[:, :k] * s[:k]) @ vt[:k, :]
    return np.clip(rebuilt, 0, 255), s

def relative_error(original, rebuilt):
    denominator = np.linalg.norm(original)
    return float(np.linalg.norm(original - rebuilt) / denominator) if denominator else 0.0

def low_rank_values(rows, cols, k):
    return rows * k + k + k * cols

uploaded = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to explore SVD compression.")
    st.stop()

try:
    rgb, original_size = load_image(uploaded)
except Exception as exc:
    st.error(f"Unable to read image: {exc}")
    st.stop()

rows, cols, _ = rgb.shape
st.caption(f"Original dimensions: {original_size[0]} × {original_size[1]} px · Working dimensions: {cols} × {rows} px · Maximum working side: 256 px")

mode = st.radio(
    "Choose the processing mode",
    ["Main project: Grayscale compression", "Extension: RGB colour compression"],
    horizontal=False,
)
st.write(
    "The main project converts the full image to grayscale and applies SVD to one brightness matrix. "
    "The RGB option is an extension that applies the same SVD method independently to the Red, Green and Blue channels."
)

if mode == "Main project: Grayscale compression":
    st.header("Main project — Grayscale image compression")
    gray = np.asarray(Image.fromarray(rgb.astype(np.uint8)).convert("L"), dtype=np.float64)
    max_k = min(gray.shape)
    k = st.slider("Number of singular values to keep (k)", 1, max_k, min(30, max_k),
                  help="Lower k gives stronger compression but loses more image detail.")
    rebuilt, singular_values = svd_reconstruct(gray, k)
    rebuilt_u8 = rebuilt.astype(np.uint8)
    error = relative_error(gray, rebuilt)
    original_count = rows * cols
    factor_count = low_rank_values(rows, cols, k)
    reduction = 100 * (1 - factor_count / original_count)

    left, right = st.columns(2)
    with left:
        st.image(gray.astype(np.uint8), caption="Grayscale input", use_container_width=True)
    with right:
        st.image(rebuilt_u8, caption=f"Reconstructed grayscale image (k = {k})", use_container_width=True)

    a, b, c = st.columns(3)
    a.metric("Components retained", f"{k} / {max_k}")
    b.metric("Relative reconstruction error", f"{error:.4f}")
    c.metric("Estimated matrix-value reduction", f"{reduction:.1f}%")
    st.caption("The reduction compares matrix entries with low-rank factor values. It is not a guaranteed JPG/PNG file-size reduction.")

    with st.expander("Mathematics used in the main project"):
        st.latex(r"A = U\\Sigma V^T")
        st.write("A is the grayscale pixel matrix. U and V contain directions/patterns, while Σ contains singular values ordered from largest to smallest.")
        st.latex(r"A_k = U_k\\Sigma_k V_k^T")
        st.write("We keep only the first k components. These capture the strongest patterns in the image and are used to reconstruct an approximation.")

    with st.expander("How does changing k affect the result?"):
        ks = sorted(set([1, min(5, max_k), min(10, max_k), min(20, max_k),
                         min(50, max_k), min(100, max_k), k, max_k]))
        errs, reductions = [], []
        for candidate in ks:
            approximation, _ = svd_reconstruct(gray, candidate)
            errs.append(relative_error(gray, approximation))
            reductions.append(100 * (1 - low_rank_values(rows, cols, candidate) / original_count))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ks, errs, marker="o")
        ax.set_xlabel("Singular values retained (k)")
        ax.set_ylabel("Relative reconstruction error")
        ax.set_title("Reconstruction error as k changes")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

elif mode == "Extension: RGB colour compression":
    st.header("Extension — RGB colour image compression")
    st.write("This optional extension keeps the image in colour by splitting it into three matrices and applying SVD to each one.")
    max_k = min(rows, cols)
    k = st.slider("Number of singular values to keep in each channel (k)", 1, max_k, min(30, max_k))
    channels = []
    for channel in range(3):
        rebuilt, _ = svd_reconstruct(rgb[:, :, channel], k)
        channels.append(rebuilt)
    reconstructed = np.stack(channels, axis=2)
    error = relative_error(rgb, reconstructed)
    original_count = 3 * rows * cols
    factor_count = 3 * low_rank_values(rows, cols, k)
    reduction = 100 * (1 - factor_count / original_count)

    left, right = st.columns(2)
    with left:
        st.image(rgb.astype(np.uint8), caption="Original colour image", use_container_width=True)
    with right:
        st.image(reconstructed.astype(np.uint8), caption=f"Reconstructed colour image (k = {k} per channel)", use_container_width=True)

    a, b, c = st.columns(3)
    a.metric("Components per channel", f"{k} / {max_k}")
    b.metric("Relative reconstruction error", f"{error:.4f}")
    c.metric("Estimated matrix-value reduction", f"{reduction:.1f}%")
    st.caption("This is an optional extension of the grayscale method. Value-count reduction is not the same as saved image-file size.")

with st.expander("Small mathematical demonstration — eigenvalues and SVD"):
    st.write(
        "Eigenvalues are included here only to show their mathematical connection with SVD. "
        "They are not used for face matching in this project."
    )
    st.latex(r"A^T A v_i = \\lambda_i v_i")
    st.latex(r"\\sigma_i = \\sqrt{\\lambda_i}")
    st.write(
        "For a matrix A, the eigenvalues of AᵀA are related to the singular values of A. "
        "A larger eigenvalue corresponds to a larger singular value, which represents a stronger component. "
        "SVD keeps the largest singular values when making a low-rank approximation."
    )
    st.info("This is a concept demonstration only; the app does not perform face detection, face matching, or identity recognition.")

st.markdown("---")
st.subheader("Project conclusion")
st.write(
    "The main project demonstrates how Singular Value Decomposition can approximate a grayscale image "
    "using fewer matrix values. The RGB mode is an optional extension, and the eigenvalue section briefly "
    "explains the mathematical relationship behind SVD."
)
st.caption("Educational prototype • Python, NumPy, Pillow, Matplotlib and Streamlit")
