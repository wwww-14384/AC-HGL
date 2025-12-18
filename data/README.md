## Usage
1. Install dependencies.
- pandas == 1.5.3
- torch == 1.11.0

2. Install [Qlib](github.com/microsoft/qlib). We have minimized the reliance on Qlib, and you can simply install it by
- <code>pip install pyqlib </code>
- pylib == 0.9.1.99

3. Download data from [OneDrive link](https://1drv.ms/f/c/652674690cc447e6/Eu8Kxv4xxTFMtDQqTW0IU0UB8rnpjACA5twMi8BA_PfbSA?e=ooc0za) (or an alternative choice [MEGA link](https://mega.nz/file/4OE0jK4I#h-LG7OjFnncbL_YGoSx5c0W604OdFMgALTYFcoDvgfw)) and unpack it into <code> data/ </code>.

5. Run main.py.

6. We provide two trained models: <code> model/csi300master_0.pkl, model/csi800master_0.pkl</code>

## Dataset
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
