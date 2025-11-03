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
Opens a Gradio UI in your browser to upload images and see predictions.

Notes: images are resized to 128 (CNN) or 224 (ViT-tiny). Tune via flags.


