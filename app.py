import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="SVD & Eigenfaces Lab", page_icon="🧮", layout="wide")

st.title("SVD Image Compression & Eigenfaces")
st.caption(
    "Explore how Linear Algebra can keep the main visual information in an image "
    "while using fewer numbers to represent it."
)

st.sidebar.header("Project modules")
module = st.sidebar.radio("Choose a module", ["SVD Image Compression", "Eigenfaces (concept demo)"])


def image_to_gray_matrix(image: Image.Image, max_side: int = 256):
    """Convert an image to grayscale and resize it for a fast demo."""
    image = ImageOps.exif_transpose(image).convert("L")
    image.thumbnail((max_side, max_side))
    return np.asarray(image, dtype=np.float64)


def frobenius_error(original: np.ndarray, reconstructed: np.ndarray):
    """Relative difference between original and reconstructed matrices."""
    denominator = np.linalg.norm(original, ord="fro")
    if denominator == 0:
        return 0.0
    return float(np.linalg.norm(original - reconstructed, ord="fro") / denominator)


def matrix_storage_values(rows: int, cols: int, k: int):
    """Count numeric values in a full matrix versus its rank-k factors."""
    original = rows * cols
    factors = rows * k + k + k * cols
    return original, factors


if module == "SVD Image Compression":
    st.header("1. SVD-based image compression")
    st.write(
        "In simple terms: the app looks for the strongest patterns in an image and tries to "
        "rebuild the image using only the most useful ones. You can change how many patterns "
        "it keeps and see the trade-off between image detail and the amount of information stored."
    )

    with st.expander("What is happening behind the scenes?"):
        st.markdown(
            """
            1. **Open the image:** the JPG/PNG file is decoded into pixels.
            2. **Make it simpler to process:** the app converts it to grayscale and resizes it.
            3. **Apply SVD:** the matrix is separated into components that contribute different amounts to the image.
            4. **Keep the top `k` components:** the largest singular values and their matching vectors are retained.
            5. **Rebuild the image:** the selected components are combined to make an approximation of the original.
            """
        )
        st.latex(r"A = U\\Sigma V^T")
        st.latex(r"A_k = U_k\\Sigma_kV_k^T")
        st.caption(
            "The first formula describes the full SVD decomposition. The second describes rebuilding the image "
            "with only the first k components."
        )

    uploaded = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Start by uploading an image. A portrait or a detailed landscape works well.")
        st.stop()

    raw_bytes = uploaded.getvalue()
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        original_size = image.size
        matrix = image_to_gray_matrix(image)
    except Exception as exc:
        st.error(f"Could not read this image: {exc}")
        st.stop()

    rows, cols = matrix.shape
    actual_file_mb = len(raw_bytes) / (1024 * 1024)
    st.write(
        f"**File:** {uploaded.name} · **File size:** {actual_file_mb:.3f} MiB · "
        f"**Original dimensions:** {original_size[0]} × {original_size[1]} pixels · "
        f"**Working matrix:** {rows} × {cols} grayscale pixels"
    )
    st.caption(
        "To keep the demo quick, the working image is resized to at most 256 × 256 pixels. "
        "The file size above is the actual uploaded file size; the calculations below compare matrix representations, "
        "not a guaranteed smaller JPG/PNG file."
    )

    try:
        u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        st.error("SVD could not finish on this image. Please try a different image.")
        st.stop()

    max_k = len(singular_values)
    default_k = min(30, max_k)
    k = st.slider(
        "How many components should we keep? (k)",
        min_value=1,
        max_value=max_k,
        value=default_k,
        help="Small k keeps fewer patterns; large k keeps more detail.",
    )

    reconstructed = (u[:, :k] * singular_values[:k]) @ vt[:k, :]
    reconstructed = np.clip(reconstructed, 0, 255)
    error = frobenius_error(matrix, reconstructed)
    original_values, factor_values = matrix_storage_values(rows, cols, k)
    storage_percent = 100 * factor_values / original_values
    reduction_percent = 100 * (1 - factor_values / original_values)

    st.subheader("Compare the images")
    c1, c2 = st.columns(2)
    with c1:
        st.image(matrix.astype(np.uint8), clamp=True, use_container_width=True, caption="Original image after grayscale conversion and resizing")
    with c2:
        st.image(reconstructed.astype(np.uint8), clamp=True, use_container_width=True, caption=f"Rebuilt image using k = {k}")

    st.info(
        f"**In everyday language:** the app kept {k} of the available {max_k} components to rebuild this image. "
        f"Move the slider up to keep more detail, or down to keep fewer components."
    )

    st.subheader("Results and what they mean")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Working image size", f"{rows} × {cols}", help="Number of grayscale pixel values used in the calculation.")
    m2.metric("Components kept", f"{k} of {max_k}", help="Think of these as the image patterns retained for rebuilding.")
    m3.metric("Relative reconstruction error", f"{error:.4f}", help="Lower generally means the rebuilt image is closer to the working original.")
    m4.metric("Estimated reduction in value count", f"{reduction_percent:.1f}%", help="An estimate of fewer numeric values needed for the low-rank factors, not the actual JPG/PNG file-size reduction.")

    with st.expander("How should I interpret these numbers?"):
        st.markdown(
            f"""
            - **Working image size ({rows} × {cols}):** the grayscale image is represented by a grid of pixel brightness values.
            - **Components kept ({k} of {max_k}):** `k` controls how many of the strongest components are used to rebuild the image.
            - **Reconstruction error ({error:.4f}):** this measures how different the rebuilt pixel matrix is from the working original. A value closer to zero means a closer match.
            - **Estimated value-count reduction ({reduction_percent:.1f}%):** compares the number of values in the full matrix with the number of values in the smaller factor matrices. It is not a promise that a saved JPG/PNG will shrink by this percentage.
            """
        )

    st.subheader("What changes when k changes?")
    st.write(
        "This graph compares image difference with the estimated reduction in the number of values. "
        "Usually, keeping more components makes the reconstructed image closer to the original, but the representation needs more values."
    )

    candidate_ks = sorted(
        set([1, min(5, max_k), min(10, max_k), min(20, max_k), min(50, max_k), min(100, max_k), k, max_k])
    )
    errors, reductions = [], []
    for candidate in candidate_ks:
        approximation = (u[:, :candidate] * singular_values[:candidate]) @ vt[:candidate, :]
        errors.append(frobenius_error(matrix, approximation))
        _, count = matrix_storage_values(rows, cols, candidate)
        reductions.append(100 * (1 - count / original_values))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(candidate_ks, errors, marker="o", label="Reconstruction error")
    ax1.set_xlabel("Number of components kept (k)")
    ax1.set_ylabel("Reconstruction error (lower is closer)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(candidate_ks, reductions, marker="s", linestyle="--", label="Estimated value-count reduction")
    ax2.set_ylabel("Estimated reduction in value count (%)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("How are eigenvalues and eigenvectors connected to SVD?"):
        st.markdown(
            """
            Imagine checking which directions or patterns are important in the image.

            - **Eigenvectors** describe those directions or patterns.
            - **Eigenvalues** indicate how strongly each direction contributes.
            - **Singular values** play a closely related role in SVD.
            """
        )
        st.latex(r"A^T A v_i = \\lambda_i v_i")
        st.latex(r"\\sigma_i = \\sqrt{\\lambda_i}")
        st.write(
            "For a matrix A, the right singular vectors are eigenvectors of AᵀA, and the eigenvalues of AᵀA "
            "are the squared singular values. This is why keeping the largest singular values retains the strongest components. "
            "The app computes SVD directly rather than building AᵀA explicitly."
        )

else:
    st.header("2. Eigenfaces (concept demo)")
    st.write(
        "Eigenfaces is the second part of the project. Instead of studying just one image, we look at several face images together. "
        "The app finds the average face and the main ways the uploaded faces differ from that average."
    )

    with st.expander("What happens in this module?"):
        st.markdown(
            """
            1. Upload several face images.
            2. Convert them to the same grayscale size so they can be compared fairly.
            3. Calculate the average face.
            4. Subtract the average from each face to focus on differences.
            5. Use SVD to find the main directions of variation. Display these directions as Eigenfaces.
            6. Represent each face using its component scores and find the closest other uploaded image.
            """
        )

    files = st.file_uploader(
        "Upload several face images (use filenames such as alex_1.jpg, alex_2.jpg, sam_1.jpg)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="faces",
    )
    if not files or len(files) < 3:
        st.info("Upload at least 3 face images. For a useful comparison, include multiple photos of each person.")
        st.stop()

    target_size = (64, 64)
    vectors, labels, previews, valid_names = [], [], [], []
    for f in files:
        try:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(f.getvalue()))).convert("L")
            im = ImageOps.fit(im, target_size)
            arr = np.asarray(im, dtype=np.float64) / 255.0
            vectors.append(arr.flatten())
            labels.append(Path(f.name).stem.rsplit("_", 1)[0])
            previews.append(arr)
            valid_names.append(f.name)
        except Exception:
            st.warning(f"Skipping unreadable file: {f.name}")

    if len(vectors) < 3:
        st.error("Not enough readable images.")
        st.stop()

    X = np.vstack(vectors)
    mean_face = X.mean(axis=0)
    centered_faces = X - mean_face
    _, singular_values, vt = np.linalg.svd(centered_faces, full_matrices=False)
    max_components = max(1, min(len(vectors) - 1, len(singular_values)))
    k_face = st.slider(
        "How many Eigenfaces/components should we keep?",
        min_value=1,
        max_value=max_components,
        value=min(5, max_components),
        key="k_face",
    )
    basis = vt[:k_face, :]
    weights = centered_faces @ basis.T

    st.subheader("Average face")
    st.caption("This is the average pixel brightness pattern across all uploaded face images.")
    st.image(mean_face.reshape(target_size), clamp=True, width=220)

    st.subheader("Main Eigenfaces")
    st.caption("These are visual patterns that capture ways the uploaded faces differ. They are not separate people.")
    display_cols = st.columns(min(k_face, 5))
    for i in range(min(k_face, 5)):
        eigenface = basis[i].reshape(target_size)
        low, high = float(eigenface.min()), float(eigenface.max())
        display = (eigenface - low) / (high - low) if high > low else np.zeros_like(eigenface)
        with display_cols[i]:
            st.image(display, clamp=True, caption=f"Component {i+1}", use_container_width=True)

    st.write(
        "Each face is now represented by a short list of component scores instead of comparing every pixel directly. "
        "The app compares those lists and finds the closest image in the uploaded collection."
    )
    query_index = st.selectbox(
        "Choose an uploaded image to test",
        list(range(len(valid_names))),
        format_func=lambda i: valid_names[i],
    )
    distances = np.linalg.norm(weights - weights[query_index], axis=1)
    distances[query_index] = np.inf
    match_index = int(np.argmin(distances))
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(previews[query_index], clamp=True, caption=f"Test image: {valid_names[query_index]}", width=220)
    with col_b:
        st.image(previews[match_index], clamp=True, caption=f"Closest other image: {valid_names[match_index]}", width=220)

    same_label = labels[query_index] == labels[match_index]
    st.write(f"**Filename label for test image:** `{labels[query_index]}`")
    st.write(f"**Filename label for closest image:** `{labels[match_index]}`")
    if same_label:
        st.success("The closest image has the same filename-based label. This is a useful demo result, but it is not proof of identity.")
    else:
        st.info("The closest image has a different filename-based label. Lighting, pose and image alignment can affect the result.")
    st.caption(
        "This is a small educational nearest-image comparison, not a verified identity prediction or a production face-recognition system."
    )

    st.subheader("What do the eigenvalues tell us?")
    eigenvalues = singular_values[:k_face] ** 2
    st.write(
        "The eigenvalues indicate how much variation is captured by each component. Larger values mean that a component explains more of the differences across the uploaded face images."
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(np.arange(1, len(eigenvalues) + 1), eigenvalues)
    ax.set_xlabel("Component number")
    ax.set_ylabel("Eigenvalue (proportional)")
    ax.set_title("How much variation each component captures")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.sidebar.markdown("---")
st.sidebar.caption("Educational prototype • Python, NumPy, Pillow, Matplotlib, Streamlit")
