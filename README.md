# SVD-Based Image Compression

## Project structure

1. **Main project — Grayscale image compression:** converts the uploaded image to grayscale, applies SVD to one brightness matrix, keeps the first `k` singular values, and reconstructs a grayscale approximation.
2. **Optional extension — RGB colour compression:** splits the colour image into Red, Green and Blue matrices, applies the same SVD method independently to each channel, and combines the reconstructed channels.
3. **Small eigenvalue demonstration:** explains the relationship between eigenvalues of AᵀA and singular values. There is no face matching or identity recognition.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The working image is resized so its longest side is at most 256 pixels for practical computation.

The displayed storage reduction is an estimate based on numeric matrix values versus low-rank factor values. It is not a guaranteed JPG/PNG file-size reduction, and this educational app does not export a compressed image file.
