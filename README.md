## ReLoop-3D

A minimal pipeline to classify rendered views of simple 3D meshes. Includes a Gradio app for uploading a mesh, rendering multiple views, and running a trained classifier.

### Requirements
- Windows 10/11 or macOS/Linux
- Python 3.10+
- GPU optional (PyTorch supports CPU)

### Install
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### Run the app
```bash
python -m src.app
```
What you get:
- Image tab: test the trained model on a single image.
- 3D Model tab: upload `.obj/.ply/.stl/.glb/.gltf`, see a live web preview, render N views offscreen, and get predictions aggregated over views.

Notes for 3D tab:
- Preview: the selected mesh is converted to a temporary `.glb` for robust browser viewing.
- Rendering: the same geometry is normalized and rendered offscreen. Backends are tried in order: Blender (if provided) → Open3D → pyrender → matplotlib.
- You can pass a Blender path (optional) for best visual fidelity on complex assets.

### Train a model (toy example)
This repo includes simple classifiers (CNN or ViT-tiny) trained on renderings. Adjust to your data as needed.
```bash
# CNN (image size 128)
python -m src.train --data_dir data --out_dir outputs --epochs 5 --model cnn

# ViT-tiny (image size 224)
python -m src.train --data_dir data --out_dir outputs --epochs 5 --model vit_tiny
```
The app loads `outputs/model_latest.pt` by default. To change the directory, edit the textbox at the top of the UI.

### Active Learning (Phase 2)
Phase 2 adds simple scripts for an optional feedback loop.

1) Select uncertain meshes into a queue CSV
```bash
python -m src.active_select --mesh_dir path/to/unlabeled_meshes --checkpoint outputs --num_views 4 --out_csv annotations/active_queue.csv
```
Fill the `label` column in the CSV.

2) Apply labels: render meshes into dataset folders
```bash
python -m src.active_apply --queue annotations/active_queue.csv --out_images data/images --image_size 224 --num_views 4
```
Then retrain with the updated dataset.

### Troubleshooting
- No 3D preview: ensure file is one of the supported formats. For `.gltf` with external textures, keep textures next to the file. The app converts to `.glb` for preview automatically.
- Renders are blank: try a simpler mesh first; then provide a Blender path for higher-quality fallback. The app also falls back to `pyrender` and `matplotlib` if Open3D offscreen is unavailable.
- Performance: reduce “Num Views”, or train the CNN model (128×128) instead of ViT (224×224).

### Useful scripts
- `src/view_o3d.py`: quick local viewer for one or more meshes using Open3D.
- `src/train.py`: trains a small classifier and writes `outputs/model_latest.pt`.
- `src/app.py`: Gradio UI entrypoint.

### References
- Open3D: https://github.com/isl-org/Open3D

## Citation

If you use this work, please cite:

```bibtex
@software{ReLoop3D,
  title={ReLoop3D: 3D Active Learning with Feedback for Machine Learning},
  author={Abaryan},
  year={2025},
  url={https://github.com/r-abaryan/ReLoop-3D}
}
```