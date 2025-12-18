# AC-HGL
AC-HGL:Heterogeneous Graph Representation Learning through Adaptive Correlation for Stock Movement Prediction
Here is the official code and supplementary materials for the GC-AGL model:AC-HGL is a model designed for stock price prediction. Then AC method constructs multiple correlation graphs and aggregates them to acquire the adaptive correlation representations. And HGL method aggregates different representations of heterogeneous graphs via varying feature strengths, and optimizes the dynamic weights based on the contributions of different modules.

<img width="1336" height="748" alt="image" src="https://github.com/user-attachments/assets/c5e594fd-9249-49b0-bc81-86ae15295b20" />

Our initial experiments were conducted within a complex business codebase developed based on Qlib.
The original code is comprehensive, and we will release the dataset and core code in the future.

<img width="185" height="145" alt="image" src="https://github.com/user-attachments/assets/49b3e8ed-1ee8-4b42-8ae9-ab62f4c56f2c" /> <img width="127" height="141" alt="image" src="https://github.com/user-attachments/assets/e1c45153-6089-4233-b8e7-78de0becb205" /> <img width="117" height="143" alt="image" src="https://github.com/user-attachments/assets/5a02d834-87f1-4065-9e54-fec82b421fe9" />


The box plot and Gaussian distribution of Cross-Moran’s I p-values compare significance under crisis (2008) and normal market conditions (2019). Overall, p-values cluster predominantly below 0.05, confirming stable statistical significance and robustness to market regimes.
During crisis periods, p-values exhibit a tightly concentrated and low-value distribution, reflecting strong directional impacts that make the null hypothesis easier to reject.
In contrast, normal markets show a wider and higher-median distribution, as more heterogeneous drivers (macroeconomics, industry dynamics, firm-level behavior, policy expectations) create greater variability in spatial dependence.

<img width="201" height="156" alt="image" src="https://github.com/user-attachments/assets/6dc496f6-dc48-43e3-9416-69b27e111067" /> <img width="195" height="159" alt="image" src="https://github.com/user-attachments/assets/e067b570-6bf8-4d5b-845f-05d41951c70b" />




Comparing SHAP values across regimes reveals a reversal in feature effects. For example, TS_F20 (momentum divergence) negatively drives the prediction of “panic sell-off’’ during crises but positively contributes to “trend continuation’’ in normal markets. Similarly, TS_F22 (medium-term mean reversion) becomes sharply elevated during fast downward crashes (e.g., multiple circuit breakers in 2008), signaling oversold conditions that typically support “rebound’’ predictions in stable markets.
Feature impact magnitude also varies by regime: crisis periods exhibit larger effects (e.g., TS_F62 volatility SHAP > 0.3), while normal markets show more moderate contributions (e.g., TS_F54 price-volume interaction mostly within ±0.2).
These results indicate that feature strength is market-dependent, and the relational graph must adapt accordingly rather than rely on static feature assumptions. Thus, the model offers actionable interpretability by revealing regime-specific factor effects relevant to trading decisions.
