## Project Overview

### Objective and Scope

This project analyzes the demand for pork meat per capita in Poland from 2010 to 2022 using econometric models. Specifically, we:

1. **Estimate income elasticity** – measure how changes in annual gross income affect per‑capita pork consumption.
2. **Compare three demand functions** – exponential, power, and first‑order Tornquist models.
3. **Assess model fit and interpret results** – select the best model and draw conclusions about consumption responses to income changes.

### Data and Sources

We use annual data for Poland (2010–2022) covering:

* `consumption_per_capita` – annual pork consumption per person \[kg/person]
* `income` – annual gross income per person \[PLN]

Data sources:

* Statistics Poland (GUS):

  * [Average gross monthly income](https://bdl.stat.gov.pl/bdl/metadane/cechy/2497?back=True)
  * [Poland's population](https://bdl.stat.gov.pl/bdl/metadane/podgrupy/7?back=True)

* Food and Agriculture Organization (FAO):
  * [Annual Pork Consumption in Poland](https://www.fao.org/faostat/en/#data/FBS?countries=173&elements=2141&items=2733&years=2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022&output_type=table&file_type=csv&submit=true)

### Methodology

We estimate three models:

* **Exponential model**:
  $f(x) = \exp\bigl(a + \tfrac{b}{x}\bigr)$, fitted by nonlinear least squares.
* **Power model**:
  $f(x) = a \cdot x^{b}$, fitted by nonlinear least squares.
* **Tornquist model (first order)**:
  $f(x) = \frac{a \cdot x}{x + b}$, fitted by nonlinear least squares.

Model performance is evaluated using:

* Coefficient of determination $R^2$
* Mean absolute percentage error (MAPE)
* Mean absolute error (MAE)
* Parameter significance tests (t‑statistics and p‑values)

### Project Structure

1. **Introduction** – background, objectives, data, and methodology.
2. **Computational Analysis** – estimation and comparison of the three models.
3. **Conclusions** – interpretation of results and recommendations.

### Repository Contents

* `projekt.ipynb`: Jupyter Notebook with code, results, and visualizations.
* `summary/`: Directory containing compiled analysis in .md format and plots.
* `source/data.csv`: Raw dataset.

### Reproducibility

1. Clone the repository.
2. Install required Python packages (pip install -r requirements.txt).
3. Run all cells in `projekt.ipynb`.
4. Review `summary/report.md` for detailed findings.
