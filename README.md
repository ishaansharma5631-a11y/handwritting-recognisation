# Handwriting Reader: prototype

Reads handwritten names (the Kaggle "Handwriting Recognition" forms) from an uploaded scan or from
a name you draw in the browser.

```
app/
  app.py              web server (FastAPI) + API
  static/index.html   front end: Draw / Upload / Samples tabs, confidence, "what the model sees", history
  model.py            CRNN (CNN + BiLSTM, CTC) and the shared preprocessing
  checkpoints/model.pt  trained weights (~4 MB)
  train.py            training script
  prep_real.py        turns the labelled scans into training arrays
  synth.py            synthetic handwriting generator (66 open-licence Google handwriting fonts in fonts/)
  predict.py          batch-recognise a folder, writes a CSV
  real_val.csv        109 test scans I labelled by hand, used as an extra sanity check
```

## Accuracy

Trained on 48.8k of the uploaded training scans plus 40k synthetic images (6 epochs, CPU, ~80 min).

| Set | Exact name correct | Character error rate |
|---|---|---|
| 2,500 held-out training scans (never trained on) | 82.5% | 4.3% |
| 109 test scans I labelled by hand (`real_val.csv`) | 88.1% | 2.3% |

`test_predictions.csv` has the model's reading of all 41,370 test images, with a confidence column.
The confidence is strict (product of per-step probabilities), so many correct readings sit below 0.9.

## Run it

Needs Python 3.10+. No GPU needed; one prediction takes a few milliseconds on a laptop CPU.

```bash
cd app
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                          # open http://localhost:8000
```

To get the **Samples** tab, unzip `test.zip` and point the server at the folder:

```bash
unzip ../../test.zip -d ../../test_images
python app.py --samples ../../test_images/test
```

Options: `--port 8000`, `--host 0.0.0.0` (to open it from another device), `--ckpt path/to/model.pt`.

### API

`POST /api/predict` with a multipart `file` (image) and optional `source=canvas` (crops the blank
canvas around the strokes). Returns `{"text", "confidence", "char_conf", "model_input", "ms"}`.

```bash
curl -F file=@TEST_0001.jpg http://localhost:8000/api/predict
```

## Batch predictions

```bash
python predict.py --images ../../test_images/test --out predictions.csv
```

## Retrain

```bash
python prep_real.py --images train/train --csv ../written_name_train_v2.csv --out work/real --val 2500
python synth.py --csv ../written_name_train_v2.csv --n 40000 --out work/synth.npz      # optional
python train.py --train work/real_train.npz work/synth.npz --val work/real_val.npz \
                --real-val real_val.csv --img-dir ../../test_images/test --out checkpoints --epochs 6
cp checkpoints/best.pt checkpoints/model.pt
```

With a GPU, more of the 330k labelled scans and more epochs will raise accuracy noticeably.

## Notes and limits

- The model reads the alphabet A–Z, hyphen, apostrophe and space. Lowercase writing is read as
  uppercase; digits and accents are not supported (accented letters come out unaccented or wrong).
- It is trained on single-line name crops. Whole pages or several lines need to be cropped first.
- Compared with the original `hwr.py`, the input is 32×256 instead of 64×256 and the network is
  narrower, so it trains on a CPU. The real crops are only ~30 px tall, so little detail is lost.
