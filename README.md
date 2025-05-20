# Systematic Earnings Drift Explorer

**A Python based systematic trading model to investigate post-earnings announcement drift (PEAD). Features data simulation for different market regimes, backtesting, and foundational quantitative analysis.**

---

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Purpose of the Model](#purpose-of-the-model)
3.  [Key Features](#key-features)
4.  [Development Context](#development-context)
5.  [Research Paper](#research-paper)
6.  [Repository Structure](#repository-structure)
7.  [Setup Instructions](#setup-instructions)
8.  [How to Run](#how-to-run)
9.  [Understanding the Output](#understanding-the-output)
10. [Key Learnings & Future Directions](#key-learnings--future-directions)
11. [Acknowledgements & References](#acknowledgements--references)

---

## 1. Project Overview

This repository contains the code and documentation for an exploratory quantitative trading model designed to investigate the Post-Earnings Announcement Drift (PEAD) phenomenon. PEAD refers to the observed tendency for stocks that announce significant earnings surprises to experience continued abnormal price movements in the same direction for some time after the announcement.

This project was undertaken as a learning exercise to understand the fundamentals of:
* Systematic strategy development based on market anomalies.
* Financial data acquisition and processing (with a focus on simulated data for this public version).
* Signal generation from financial events.
* Building a basic backtesting framework in Python.
* Applying foundational quantitative analysis techniques, including factor attribution and performance metric calculation.

The model is primarily built to run on **internally simulated data**, allowing for experimentation and understanding of its mechanics without requiring external API access (like Bloomberg) for basic operation. It can simulate different market regimes ('POSITIVE' and 'NEGATIVE') to observe strategy performance under varied conditions. The version used with Bloomberg contains significant enhancements and will stay private indefinitely. 

## 2. Purpose of the Model

The core purpose of this model was to serve as a practical tool to:
* Translate the academic concept of PEAD into a testable, algorithmic hypothesis.
* Gain hands-on experience with the lifecycle of a quantitative modeling project, from data handling to performance evaluation.
* Explore how different market conditions (simulated as regimes) might impact an event-driven strategy.
* Understand the importance of data integrity, point-in-time accuracy, and the challenges of backtesting.
* Introduce concepts of factor analysis to differentiate strategy alpha from market beta.


## 3. Key Features

* **Event-Driven Strategy:** Focuses on trading signals derived from earnings announcement surprises.
* **Simulated Data Engine:** Includes a `EarningsDataCollector` class capable of generating randomized, regime-aware simulated data for:
    * Earnings announcement dates.
    * Point-in-Time (PIT) actual and consensus earnings figures (EPS & Sales).
    * Daily stock market data (Open, High, Low, Close, Volume).
    * Common Fama-French style factor returns (MKT_RF, SMB, HML, MOM, QMJ, BAB) and a risk-free rate.
* **Regime Simulation:** Can simulate 'POSITIVE' and 'NEGATIVE' market regimes, influencing the characteristics of the generated data to test strategy robustness.
* **Signal Generation:** Calculates EPS surprise percentages and Standardized Unexpected Earnings (SUE) scores.
* **Backtesting Framework:** A custom event-driven backtester (`EarningsBacktester`) processes signals and simulates trades.
* **Portfolio Management:** A `Portfolio` class tracks hypothetical positions, cash, and P&L, including basic transaction cost modeling.
* **Performance Analysis:** The `PerformanceAnalyzer` calculates:
    * Basic metrics (Total Return, Annualized Return, Volatility, Sharpe Ratio, Max Drawdown, Win Rate).
    * Factor attribution analysis (Alpha, Betas to common factors, R-squared).
* **Visualization:** Generates and saves plots for:
    * Portfolio equity curve and drawdown.
    * 1-year rolling annualized return, volatility, and Sharpe ratio.
    * Factor attribution (betas).
* **Parameter Configuration:** Uses a `StrategyConfig` class for easy management of all model and simulation parameters.
* **Currency Support:** Configured for GBP (£) and a £10M initial simulated capital.

## 4. Development Context

The foundational build of this model, focusing on the core concepts of earnings surprise analysis and the initial backtesting framework, was initiated as an independent research and learning project prior to my internship at Davidson Kempner Capital Management.

During the internship, an enhanced version of this model was developed and utilized for a paper trading exercise. The specifics of this enhanced version, including any proprietary modifications, data sources used, or detailed performance results from that period, will be kept **strictly private indefinitely** due to the confidential and proprietary nature of work conducted during that professional engagement.

This public repository and the accompanying paper describe the underlying quantitative principles, architectural design, and the significant learning journey undertaken in creating this exploratory model for academic and personal development purposes.

## 5. Research Paper

A detailed paper documenting the model's design, methodology, rigorous aspects, data considerations, technical details, personal reflections on the learning process, and potential future enhancements can be found in this repository:

* **[An Exploratory Journey into Quantitative Event-Driven Modeling (PDF)](https://github.com/xas-L/Systematic-Earnings-Drift-Explorer/blob/main/docs/EvDriv_Model_paperv1.5.pdf)**
    * *Note: The paper version linked (v1.5) corresponds to the exploratory nature described in this README. The Python script in this repository is `earnings_event_model.py` (equivalent to v2.2.3 discussed previously), which includes the plotting enhancements.*

## 6. Repository Structure

SystematicEarningsDrift_Explorer:
- model: contains complete model code without plotting layer
- docs: LaTeX write-up (updated)
- notebooks: model with plots shown

## 7. Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/SystematicEarningsDrift_Explorer.git](https://github.com/your-username/SystematicEarningsDrift_Explorer.git) # Replace with your actual repo URL
    cd SystematicEarningsDrift_Explorer
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Ensure your `requirements.txt` includes `pandas`, `numpy`, `statsmodels`, `matplotlib`, and `seaborn`. If you intend to try running with actual Bloomberg data (by setting `BLOOMBERG_AVAILABLE = True` in the script), you would also need `xbbg` and a configured Bloomberg terminal with API access.

## 8. How to Run

The primary model and simulation logic is contained within `earnings_event_model.py`.

1.  **Configure Simulation (Optional):**
    * Open `earnings_event_model.py`.
    * At the top, the `BLOOMBERG_AVAILABLE` flag is set to `False` by default, ensuring the model uses internally simulated data.
    * The `if __name__ == "__main__":` block at the end of the script configures and runs two simulation scenarios: one for a 'POSITIVE' market regime and one for a 'NEGATIVE' market regime. You can adjust parameters within the `StrategyConfig` instantiations for each regime (e.g., `config_positive.start_date`, `config_negative.holding_period_days`, etc.) to explore different settings.

2.  **Execute the script:**
    ```bash
    python earnings_event_model.py
    ```

The script will:
* Print status messages to the console during data preparation and backtesting for each regime.
* Generate and save performance plots and factor attribution plots to the respective subdirectories within the `results/` folder (e.g., `results/positive_regime/`).
* Print a summary performance report for each regime to the console.

## 9. Understanding the Output

After running `earnings_event_model.py`, you will find the following in the `results/` directory (organized by regime):

* **`performance_summary_plots_[REGIME].png`**: A multi-panel plot showing:
    * Portfolio Equity Curve and Drawdown.
    * 1-Year Rolling Annualized Return and Volatility.
    * 1-Year Rolling Sharpe Ratio.
* **`factor_attribution_[REGIME].png`**: A bar chart displaying the strategy's beta (loading) to common Fama-French style factors (MKT_RF, SMB, HML, MOM, QMJ, BAB), along with the annualized alpha and R-squared of the factor regression. Significance stars are included on the bars.

The console output will also provide detailed performance metrics and factor analysis summaries for each simulated regime.

## 10. Key Learnings & Future Directions

This project served as a valuable introduction to quantitative modeling. Key takeaways included:
* The paramount importance of **data integrity**, especially point-in-time accuracy for financial data.
* The nuances in defining and quantifying a "surprise" or an "event."
* The complexities and assumptions inherent in **backtesting** (e.g., transaction costs, slippage).
* The critical role of **factor attribution** in understanding true alpha versus beta exposures.
* The iterative nature of quantitative research and model development.

Future enhancements could involve:
* Integrating more sophisticated signal generation techniques (e.g., NLP on earnings calls, machine learning).
* Developing the conceptual equity options trading layer.
* Implementing more advanced portfolio construction and dynamic risk management techniques.
* Exploring adaptability to different market regimes more deeply.

## 11. Acknowledgements & References

This exploratory project was significantly inspired by the structured approach to quantitative finance presented in texts such as:
* Kelliher, C. (2022). *Quantitative Finance With Python: A Practical Guide to Investment Management, Trading, and Financial Engineering*. Chapman & Hall/CRC.

The process also benefited from the broader academic literature on market anomalies like PEAD and factor investing.
* Bernard, V. L., & Thomas, J. K. (1989). Post-earnings-announcement drift: Delayed price response or risk premium?. *Journal of Accounting research*, 1-36.

**Disclaimer:** This repository is for educational and illustrative purposes. The model is based on simulated data for this public version. Any strategies or concepts discussed do not constitute investment advice. Copyrighted materials, such as the full text of referenced books, are not included in this repository.
