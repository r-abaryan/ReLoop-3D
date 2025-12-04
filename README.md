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
- Image → 3D tab: convert image(s) to 3D mesh.
  - Single mode: depth estimation (MiDaS) → mesh (fast).
  - Multi-view mode: Open3D point cloud fusion → mesh (better quality, no external deps).
- 3D Model tab: upload `.obj/.ply/.stl/.glb/.gltf`, see live preview, render N views, classify.

Notes:
- 3D preview converts to `.glb` for browser viewing.
- Rendering backends: Blender (optional) → Open3D → pyrender → matplotlib.
- Multi-view uses Open3D point cloud fusion (no external tools needed).

### Train a model (toy example)
This repo includes simple classifiers (CNN or ViT-tiny) trained on renderings. Adjust to your data as needed.
```bash
# CNN (image size 128)
python -m src.train --data_dir data --out_dir outputs --epochs 5 --model cnn

# ViT-tiny (image size 224)
python -m src.train --data_dir data --out_dir outputs --epochs 5 --model vit_tiny
```
The app loads `outputs/model_latest.pt` by default. To change the directory, edit the textbox at the top of the UI.

### Active Learning
Use the Active Learning tab in the UI:
1. Score Unlabeled: point to mesh folder, choose acquisition (entropy/margin), click Score.
2. Edit labels in the table.
3. Apply Labels: renders meshes into `data/images/<label>/` and logs to `annotations/labeled.csv`.
4. Retrain with the updated dataset.

CLI scripts also available: `src/active_select.py`, `src/active_apply.py`.

### Troubleshooting
- No 3D preview: ensure file is supported format. For `.gltf` with external textures, keep textures next to file.
- Renders blank: try simpler mesh first; provide Blender path for fallback.
- Image→3D poor quality (single mode): use Multi-view mode with 10–30 images orbiting the object for better results.
- Performance: reduce "Num Views", or use CNN (128×128) instead of ViT (224×224).

### Useful scripts
- `src/train.py`: train classifier, writes `outputs/model_latest.pt`.
- `src/app.py`: Gradio UI entrypoint.
- `src/depth_to_mesh.py`: single-image depth → mesh pipeline.
- `src/multiview_to_mesh.py`: Open3D multi-view point cloud fusion → mesh.
- `src/view_o3d.py`: quick local Open3D viewer for meshes.

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