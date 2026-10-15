# SVD Image Compression & Eigenfaces Demo

This version keeps the original technical outputs and adds plain-language explanations beside them.

## Run on Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If activation is blocked in PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Run on Ubuntu/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Modules
### SVD Image Compression
Upload JPG/PNG, change `k`, compare the original working grayscale image with the rank-k reconstruction, inspect reconstruction error and estimated matrix-value reduction, and view the trade-off graph. Expand the explanations for the mathematical details.

### Eigenfaces
Upload several face images with filenames such as `alex_1.jpg`, `alex_2.jpg`, `sam_1.jpg`. The app displays the average face, main Eigenfaces, nearest other uploaded image, and component eigenvalue plot.

## Important interpretation
- Images are resized to at most 256 × 256 in the SVD module for responsiveness.
- The storage metric estimates numeric values required for a full matrix versus low-rank factors. It is not a measured JPG/PNG file-size reduction.
- Eigenfaces matching is a small educational nearest-image demonstration, not verified identity recognition.
