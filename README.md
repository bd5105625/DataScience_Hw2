# DataScience_Hw2

## DataSet:

### Training Data:

* 約1500筆(天)IBM股票的開盤價、當日最高價、當日最低價和當日收盤價
### Testing Data:
* 20筆(天)的測試資料集，同時也是最後需要預測的範圍

## Data Preprocessing:

因為此次目的是判斷開盤價格來決定是否要購買或放空股票，因此利用pandas取得每個資料的開盤價格(第一個column)作為訓練資料。

## Model: LSTM

使用LSTM模型，此模型適用於時序性的資料，具有記憶功能，因此對於股票預測有很大幫助。
經測試後，利用欲預測日前十天作為輸入資料來預測該日可能的開盤價格。
訓練過程中使用30個epoch，batch size則為5，使用的loss function 為"mean absolute error"，而optimizer為Adam。
