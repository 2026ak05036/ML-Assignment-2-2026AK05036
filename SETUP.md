# Evening Execution Guide — BITS Virtual Lab

Assumes the lab has reset and you are starting from a fresh session.
Work through the stages in order. Each stage says what you should see.

Environment this was written for: Rocky Linux, Anaconda base,
Python 3.12.7, scikit-learn 1.9.1, Streamlit 1.67.0.

---

## Stage 1 — Upload and extract (2 min)

Upload `ML-Assignment-2-2026AK05036.zip` into the lab, then:

```bash
cd ~
unzip -o ML-Assignment-2-2026AK05036.zip
cd ML-Assignment-2-2026AK05036
ls -la
```

**Expect:** `app.py`, `train_models.py`, `download_data.py`,
`verify_project.py`, `requirements.txt`, `README.md`, `SETUP.md`, `.gitignore`

---

## Stage 2 — Confirm versions (1 min)

```bash
python -c "import sklearn, pandas, numpy, joblib, streamlit; print('sklearn', sklearn.__version__); print('pandas', pandas.__version__); print('numpy', numpy.__version__); print('joblib', joblib.__version__); print('streamlit', streamlit.__version__)"
```

**Expect:** `sklearn 1.9.1` and `streamlit 1.67.0`.

**If scikit-learn is NOT 1.9.1**, edit `requirements.txt` and change the
`scikit-learn==` line to the version actually printed. The models are
pickled by this version; Streamlit Cloud must install the same one or the
deployed app will fail to load them.

### joblib version note

`requirements.txt` pins `joblib==1.6.0` while the lab has 1.4.2. This is
intentional and safe: the pickle format is compatible in both directions
(verified), and joblib 1.4.2 combined with NumPy 2.5 emits dozens of
harmless DeprecationWarnings when loading models. Pinning 1.6.0 keeps the
deployed app's logs clean.

If you also want clean output in the lab:  `pip install --user joblib==1.6.0`

### About the protobuf / TensorFlow warning

Installing Streamlit upgraded protobuf, which pip flags as conflicting with
TensorFlow 2.17. **This project never imports TensorFlow**, so the warning is
harmless here and you can ignore it.

Only if something genuinely breaks, use an isolated environment:

```bash
python -m venv ~/ml2env
source ~/ml2env/bin/activate
pip install -r requirements.txt
```

Remember to re-activate it (`source ~/ml2env/bin/activate`) in every new
terminal, and re-check the version in Stage 2.

---

## Stage 3 — Download the dataset (2 min)

```bash
pip install --user ucimlrepo
python download_data.py
```

**Expect:**
```
shape           : (12330, 18)
input features  : 17
missing values  : 0
>=500 instances :  12330  [PASS]
>=12 features   :      17  [PASS]
```

**If the download is blocked:** open Firefox inside the lab, go to
`https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset`,
click Download, extract the zip, and place `online_shoppers_intention.csv`
in the project folder. Then re-run `python download_data.py` to validate.

---

## Stage 4 — Train the models (3–6 min)

```bash
python train_models.py
```

**Expect** five printed steps ending in `TRAINING COMPLETE`, and these new
files: `model/` (5 `.joblib` files + `metrics.json`) and `test_data.csv`.

**Copy the STEP 4 comparison table somewhere safe — those are the real
numbers for your README.**

Troubleshooting:
- `FileNotFoundError` → Stage 3 did not produce the CSV.
- `ValueError: does not meet the assignment minimums` → wrong CSV downloaded.
- Killed / out of memory → close other lab applications and retry.

---

## Stage 5 — Automated checks (1 min)

```bash
python verify_project.py
```

At this point expect roughly `21/25`. The remaining failures are the README
placeholders and they are expected until Stage 7.

---

## Stage 6 — Test the app locally (5 min)

```bash
streamlit run app.py
```

Open the Local URL shown. Then:

1. Upload `test_data.csv` from the project folder
2. Switch through all five models in the sidebar dropdown
3. Confirm six metrics, the confusion matrix and the classification report render
4. Scroll to the all-model comparison table
5. Check the metrics match your Stage 4 table

Stop the app with `Ctrl+C`.

Troubleshooting:
- "model files are missing" → Stage 4 did not finish.
- "must contain the target column Revenue" → wrong CSV uploaded.

---

## Stage 7 — Complete the README (20–30 min, your own work)

Open `README.md` and:

1. Paste your real Stage 4 numbers into the **comparison table** under
   section d.
2. Write the **five observations** plus the overall winner. This section
   carries 3 marks — see the interpretation guide in the chat.
3. Leave the Streamlit URL line for Stage 9.

---

## Stage 8 — Push to GitHub (10 min)

Commit in stages — the professor reviews commit history.

```bash
git init
git remote add origin https://github.com/2026ak05036/ML-Assignment-2-2026AK05036.git

git add train_models.py download_data.py requirements.txt .gitignore
git commit -m "Add training script, data loader and dependencies"

git add model/ test_data.csv
git commit -m "Add trained model pipelines and held-out test split"

git add app.py
git commit -m "Add Streamlit application for model evaluation"

git add README.md SETUP.md verify_project.py
git commit -m "Add README with results and project verification script"

git branch -M main
git push -u origin main
```

If prompted for a password, GitHub needs a **personal access token**, not
your account password: GitHub → Settings → Developer settings → Personal
access tokens → Fine-grained tokens → generate with `repo` write access.

Then open the repo in a browser and confirm `model/` contains all five
`.joblib` files.

---

## Stage 9 — Deploy to Streamlit Community Cloud (10 min)

1. Go to https://streamlit.io/cloud and sign in with GitHub
2. **New app** → **Deploy a public app from GitHub**
3. Repository: `2026ak05036/ML-Assignment-2-2026AK05036`
4. Branch: `main`
5. Main file path: `app.py`
6. Advanced settings → Python version: **3.12**
7. **Deploy**

First build takes 3–8 minutes.

Once live, test it: upload `test_data.csv`, switch models, confirm the
metrics match. Then paste the URL into `README.md`, commit and push.

Troubleshooting (open **Manage app** → logs):
- `ModuleNotFoundError` → package missing from `requirements.txt`
- `InconsistentVersionWarning` or unpickling error → the `scikit-learn` pin
  does not match the version that trained the models
- `FileNotFoundError: model/...` → the `model/` folder was not pushed; check
  that `.gitignore` is not excluding it

---

## Stage 10 — BITS Lab screenshot (2 min)

In the lab, run:

```bash
python train_models.py
python verify_project.py
```

Take **one screenshot** showing the output on screen with the BITS Virtual
Lab window and URL visible. Save as PNG or PDF.

---

## Stage 11 — Final checks and PDF (15 min)

```bash
python verify_project.py
```

**Must now print `ALL CHECKS PASSED`.** If not, fix what it reports.

Build a single PDF in exactly this order:

1. GitHub repository link (clickable)
2. Live Streamlit app link (clickable)
3. BITS Virtual Lab screenshot
4. The full README content

Upload to Taxila under Assignment 1 and **click SUBMIT** — drafts are not
accepted. Deadline: 8 November 2026, 23:59.
