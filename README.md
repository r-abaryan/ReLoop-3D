## 3D Active Learning (Toy Demo)

Minimal PyTorch pipeline to classify rendered 3D shapes and improve via an active-learning loop.

### Install
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

### Data
- Labeled: either `data/labels.csv` (columns: `image_path,label`) or folders `data/images/<class>/*.png`
- Unlabeled pool: `data/unlabeled/*.png`

### Train
```bash
python -m src.train --data_dir data --out_dir outputs --epochs 5 --model cnn
# or a small transformer (ViT-tiny, image_size defaults to 224)
python -m src.train --data_dir data --out_dir outputs --epochs 5 --model vit_tiny
```

### Active Learning (select uncertain unlabeled)
```bash
python -m src.active_learning --data_dir data --out_dir outputs --num_query 50
```
Outputs `annotations/annotation_queue.csv`. Fill the `label` column.

### Retrain with Corrections
```bash
python -m src.retrain_with_corrections --data_dir data --out_dir outputs --corrections annotations/corrections.csv --epochs 3
```
Merges labels, moves images into `data/images/<label>/`, and continues training from the latest checkpoint.

### Evaluate & Visualize
```bash
python -m src.eval --data_dir data --out_dir outputs
```
Saves `outputs/confusion_matrix.png` and `outputs/eval_metrics.csv`.

### Single-Image Test
```bash
python -m src.predict --image path/to/image.png --out_dir outputs
```
Prints top predictions and saves `outputs/prediction_annotated.png`.

### Playground (UI)
```bash
python -m src.app
```
Opens a Gradio UI with two tabs:
- Image: upload a single image and see predictions.
- 3D Model: upload a simple mesh (.obj/.ply/.stl). The app renders multiple offscreen views via Open3D and aggregates predictions across views.
  - If Open3D headless isn’t available on Windows, it falls back to pyrender+trimesh. You can also provide a Blender path for GPU-quality renders.
 - Web Preview (Plotly): interactive, browser-based mesh viewer (no OpenGL needed). Useful when headless rendering isn’t available; not used for training snapshots.

Notes: images are resized to 128 (CNN) or 224 (ViT-tiny). Tune via flags.

### 3D Notes
- Requires `open3d` (added in `requirements.txt`).
- Supported mesh formats tested: `.obj`, `.ply`, `.stl`.
- The 3D tab renders a few azimuths around the model; predictions are averaged across views.
 - Windows headless rendering: If you see EGL errors, the app automatically tries pyrender. For best results with NVIDIA GPUs, set the Blender path in the UI or export `BLENDER_PATH` environment variable to a `blender.exe` and the app will render via Blender in headless mode.


