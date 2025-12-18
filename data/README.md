## Usage1 (The raw dataset version without Qlib preprocessing)
1. Install dependencies.
- pandas == 1.5.3
- torch == 1.11.0

2. Install [Qlib](github.com/microsoft/qlib). We have minimized the reliance on Qlib, and you can simply install it by
- <code>pip install pyqlib </code>
- pylib == 0.9.1.99

3. Download data from [OneDrive link](https://1drv.ms/f/c/a596b5061052f949/IgDOo3vuQjwBTL8P3pd719xwAc8Ju0RA9gBnJuQV_dI27Hs?e=cBJE8J)
   
4. Run main.py.

## Dataset1
### Form
The downloaded data is split into training, validation, and test sets, with two stock universes. Note the csi300 data is a subset of the csi800 data.
You can use the following code to investigate the **datetime, instrument, and feature formulation**.
```python
with open(f'data/csi300/csi300_dl_train.pkl', 'rb') as f:
    dl_train = pickle.load(f)
    dl_train.data # a Pandas dataframe
```
In our code, the data will be gathered chronically and then grouped by prediction dates. the <code> data </code> iterated by the data loader is of shape (N, T, F), where:
- N - number of stocks. For CSI300, N is around 300 on each prediction date; For CSI800, N is around 800 on each prediction date.
- T - length of lookback_window, T=8.
- F - 222 in total, including 158 factors, 63 market information, and 1 label.   

## Usage2 (the complete version of the script)

### config

> Note that you should install `torch` and by your self.
```
bash config.sh
```
### run
```
conda activate ACHGL
bash run.sh
```
<!-- or you can just directly use `qrun` tp run the codes (note that you should modify your `qlib`, since we add or modify some files in `qlib/contrib/data/dataset.py`, `qlib/data/dataset/__init__.py`, `qlib/data/dataset/processor.py` and `qlib/contrib/model/pytorch_achgl.py`):
```
qrun workflow_config_achgl_Alpha158.yaml
```-->
### Note
```
-The complete scripts will be released upon the official acceptance of the paper.
The initial version has verified the potential of the model, and additional test scripts (e.g., for noise robustness testing) will be supplemented in the future.-

