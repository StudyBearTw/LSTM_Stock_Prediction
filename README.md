# 📈 飆股預測系統 (Hot Stock Prediction using LSTM)

本專案旨在建立一套基於 LSTM 的飆股預測模型，透過籌碼面、價量面與基本面等資料，預測未來 30 天內可能漲幅超過 10% 的股票。

---

## 🧠 專案簡介

- **模型目標**：判斷某股票是否為「飆股」（定義：未來 30 天內漲幅 ≥ 10%）
- **使用模型**：LSTM（長短期記憶神經網路）
- **核心特徵類別**：
  - 籌碼面：外資買賣超、融資、融券等
  - 價量面：KD 指標、報酬率、成交量等技術指標
  - 基本面：營收、EPS、營收成長率等

---

## ⚙️ 模型訓練流程

1. **資料預處理**
   - 特徵選擇與工程（籌碼、價量、基本面 + 技術指標）
   - 缺失值處理與標準化（使用 MinMaxScaler）
   - 時間序列切分（TimeWindow 滑動窗口）
   - 類別不平衡處理（SMOTE 重抽樣）

2. **模型建構與訓練**
   - LSTM 模型架構（包含 Dropout、EarlyStopping、ReduceLROnPlateau）
   - 損失函數：Binary Crossentropy
   - 評估指標：Precision、Recall、F1-score

3. **預測與最佳閾值尋找**
   - 使用 F1-score 選擇最適合的分類閾值
   - 比較不同 threshold 下的表現差異

---

## ✅ 模型最終成果

| 指標        | 數值     |
|-------------|----------|
| Precision   | 0.7573   |
| Recall      | 0.8545   |
| **F1 Score**| **0.8029** |

> 🎯 模型對於飆股預測有良好的召回率與整體表現，適用於高風險偏好之選股輔助分析。

---

## 🔮 未來展望

- 引入 Attention 機制，強化關鍵時間序列特徵
- 融合多模型（如 LSTM + XGBoost）進行預測
- 加入 SHAP 值解釋模型預測結果，提升可解釋性
- 與即時資料串接，進行每日預測與回測驗證

---

## 📌 技術環境

- Python 3.10
- TensorFlow / Keras
- pandas, numpy, scikit-learn
- imbalanced-learn（SMOTE）
- talib, ta（技術指標）

---

## 📬 聯絡資訊

如需合作或技術討論，歡迎聯繫：

- 作者: StudyBearTW

# 📈 Hot Stock Prediction System (Using LSTM)

This project aims to build a hot stock prediction model based on LSTM, using data from the market sentiment, price-volume indicators, and fundamentals to predict stocks that are likely to increase by more than 10% in the next 30 days.

---

## 🧠 Project Overview

- **Model Objective**: Predict whether a stock is a "hot stock" (defined as having a price increase of ≥ 10% in the next 30 days)
- **Model Used**: LSTM (Long Short-Term Memory Neural Network)
- **Key Feature Categories**:
  - Market Sentiment: Foreign institutional trading, margin trading, short selling, etc.
  - Price-Volume Indicators: KD indicators, returns, trading volume, and other technical indicators
  - Fundamentals: Revenue, EPS, revenue growth rate, etc.

---

## ⚙️ Model Training Process

1. **Data Preprocessing**
   - Feature selection and engineering (market sentiment, price-volume, fundamentals + technical indicators)
   - Handling missing values and standardization (using MinMaxScaler)
   - Time series split (TimeWindow sliding window)
   - Addressing class imbalance (SMOTE resampling)

2. **Model Construction and Training**
   - LSTM model architecture (including Dropout, EarlyStopping, ReduceLROnPlateau)
   - Loss function: Binary Crossentropy
   - Evaluation metrics: Precision, Recall, F1-score

3. **Prediction and Optimal Threshold Selection**
   - Select the best classification threshold using F1-score
   - Compare performance under different thresholds

---

## ✅ Final Model Performance

| Metric      | Value    |
|-------------|----------|
| Precision   | 0.7573   |
| Recall      | 0.8545   |
| **F1 Score**| **0.8029** |

> 🎯 The model demonstrates good recall and overall performance for hot stock prediction, suitable for high-risk stock selection assistance.

---

## 🔮 Future Directions

- Incorporate Attention Mechanism to enhance key time-series features
- Combine multiple models (e.g., LSTM + XGBoost) for prediction
- Integrate SHAP values to explain model predictions and enhance interpretability
- Connect with real-time data for daily prediction and backtesting

---

## 📌 Technical Environment

- Python 3.10
- TensorFlow / Keras
- pandas, numpy, scikit-learn
- imbalanced-learn (SMOTE)
- talib, ta (technical indicators)

---

## 📬 Contact Information

For collaboration or technical discussions, feel free to contact:

- Author: StudyBearTW

