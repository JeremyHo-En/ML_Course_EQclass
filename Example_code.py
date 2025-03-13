import datetime
import time
import glob
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.signal
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import acf
import obspy
import obspy.signal.filter
from obspy import Trace, Stream
from obspy.core import UTCDateTime
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

#特徵工程原始碼

def Feature_Engineering(trace, Hz=100):

    #將訊號濾波
    trace_bp1 = obspy.signal.filter.bandpass(trace, 0.1, 1, 100, corners=4)
    trace_bp2 = obspy.signal.filter.bandpass(trace, 2, 8, 100, corners=4)
    trace_bp3 = obspy.signal.filter.bandpass(trace, 5, 20, 100, corners=4)
    #將訊號包絡化
    env_trace = obspy.signal.filter.envelope(trace)
    env_trace_bp1 = obspy.signal.filter.envelope(trace_bp1)
    env_trace_bp2 = obspy.signal.filter.envelope(trace_bp2)
    env_trace_bp3 = obspy.signal.filter.envelope(trace_bp3)
    #特徵萃取
    F_H_01 = np.nanmean(env_trace)/np.nanmax(env_trace)
    F_H_02 = np.nanmedian(env_trace)/np.nanmax(env_trace)
    F_H_03 = np.nanmax(env_trace)
    F_H_04 = (np.argmax(env_trace)-1)/Hz
    F_H_05 = kurtosis(trace)
    F_H_06 = kurtosis(env_trace)
    F_H_07 = skew(trace)
    F_H_08 = skew(env_trace)
    F_H_09 = np.sum(np.abs(acf(trace, nlags=100))[0:33])
    F_H_10 = np.sum(np.abs(acf(trace, nlags=100))[34:])
    F_H_11 = F_H_09/F_H_10
    F_H_12 = np.nanmax(env_trace_bp2)
    F_H_13 = len(find_peaks(env_trace_bp2))
    F_H_14 = np.nansum(env_trace_bp1)
    F_H_15 = np.nansum(env_trace_bp2)
    F_H_16 = np.nansum(env_trace_bp3)
    F_H_17 = kurtosis(env_trace_bp1)
    F_H_18 = kurtosis(env_trace_bp2)
    F_H_19 = kurtosis(env_trace_bp3)

    F_Group1 = [F_H_01, F_H_02, F_H_03, F_H_04, F_H_05, F_H_06, F_H_07, F_H_08,
               F_H_09, F_H_10, F_H_11, F_H_12, F_H_13, F_H_14, F_H_15, F_H_16,
               F_H_17, F_H_18, F_H_19]

    # The bp_buterworth routine is vectorized
    p     = 2
    order = 4
    bp = [[0.1, 0.2],
          [0.2, 1.0],
          [1.0, 5.0],
          [5.0, 10.0],
          [10.0, 20.0]]

    idx = 0
    tr_filt_0 = bp_butterworth(trace, Hz, p, order, bp[idx][0], bp[idx][1])

    idx = 1
    tr_filt_1 = bp_butterworth(trace, Hz, p, order, bp[idx][0], bp[idx][1])

    idx = 2
    tr_filt_2 = bp_butterworth(trace, Hz, p, order, bp[idx][0], bp[idx][1])

    idx = 3
    tr_filt_3 = bp_butterworth(trace, Hz, p, order, bp[idx][0], bp[idx][1])

    idx = 4
    tr_filt_4 = bp_butterworth(trace, Hz, p, order, bp[idx][0], bp[idx][1])

    # Add or remove the filtered waveforms
    trace_calcs = np.vstack([
                             tr_filt_0,
                             tr_filt_1,
                             tr_filt_2,
                             tr_filt_3,
                             tr_filt_4
                            ])

    zeroCrossRate = np.nansum(np.diff(np.sign(trace_calcs), axis=1) == 0, axis=1)

    q5,  q95 = np.nanquantile(trace_calcs, (0.05, 0.95), axis=1).astype(np.float32)
    q5_95    = q95-q5

    q10, q90 = np.nanquantile(trace_calcs, (0.10, 0.90), axis=1).astype(np.float32)
    q10_90   = q90-q10

    q25, q75 = np.nanquantile(trace_calcs, (0.25, 0.75), axis=1).astype(np.float32)
    q25_75   = q75-q25

    q40, q60 = np.nanquantile(trace_calcs, (0.40, 0.60), axis=1).astype(np.float32)
    q40_60   = q60-q40

    minmax_rng = np.nanmax(trace_calcs, axis=1).astype(np.float32)-np.min(trace_calcs, axis=1).astype(np.float32)
    t_rms = rms(trace_calcs).astype(np.float32)

    # Envelope energy
    env_energy = np.log10(np.trapz(envelope(trace_calcs), dx=1./Hz, axis=1))


    # FFT PSP vectorized
    delta = 1./Hz
    M, N = trace_calcs.shape
    N_half = int(N/2)
    amp = smooth(np.abs((np.fft.fft(trace_calcs)/N)[:, :N_half]))
    freq = np.fft.fftfreq(N, d=delta)[:N_half]
    # set up vector for vectorized calcs
    freq2d = freq.repeat(M).reshape(freq.size, M)
    max_amp_frequency = freq[np.argmax(amp, axis=1)]

    # Get the centroid and quartile info with vectorized calcs.
    # This requires masking / booleans
    fftCenter = np.sum(freq * amp, axis=1) / np.sum(amp, axis=1)
    fftBandwidth = np.sqrt( np.sum((freq2d - fftCenter).T**2 * amp, axis=1) / np.sum(amp, axis=1) )


    ################################################################
    # Fill the array in the correct order
    features_out = np.hstack([
        F_Group1,
        zeroCrossRate,
        q5_95,
        q10_90,
        q25_75,
        q40_60,
        minmax_rng,
        t_rms,
        env_energy,
        fftCenter,
        fftBandwidth
        ]).astype(np.float32)

    # features_out = F_HaoYu
    #print('Done')
    return features_out


def normalize(x):
    return x / np.max(np.abs(x))


def toYearFraction(date):
    # return the year fraction as year plus decimal
    def sinceEpoch(date):
        # returns seconds since epoch
        return time.mktime(date.timetuple())

    year = date.year
    startOfThisYear = datetime.datetime(year=year, month=1, day=1)
    startOfNextYear = datetime.datetime(year=year+1, month=1, day=1)

    yearElapsed  = sinceEpoch(date) - sinceEpoch(startOfThisYear)
    yearDuration = sinceEpoch(startOfNextYear) - sinceEpoch(startOfThisYear)
    fraction = yearElapsed/yearDuration
    return float(date.year) + fraction


def bp_butterworth(trace, sps, p, order, cnr1, cnr2):
    '''
    Butterworth Bandpass filter
    Inputs:
    trace - data
    sps   - samples per second (Hz)
    p     - pass, if two then do two pass, else do one pass
    order - filter order, number of poles
    cnr1  - low corner
    cnr2  - high corner

    This matches the sac filter results
    '''

    f_Ny = sps / 2.
    sos = scipy.signal.butter(order, [cnr1/f_Ny, cnr2/f_Ny], btype='bandpass', output='sos')
    if p == 2:
        firstpass = scipy.signal.sosfilt(sos, trace)
        return scipy.signal.sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return scipy.signal.sosfilt(sos, trace)


def envelope(data):
    """
    Envelope of a function. -- From obspy

    Computes the envelope of the given function. The envelope is determined by
    adding the squared amplitudes of the function and it's Hilbert-Transform
    and then taking the square-root. (See [Kanasewich1981]_)
    The envelope at the start/end should not be taken too seriously.

    :type data: numpy.ndarray
    :param data: Data to make envelope of.
    :return: Envelope of input data.
    """
    # hilb = Hilbert(data)
    hilb = np.imag(scipy.signal.hilbert(data, axis=1)) # The fftpack returns the imaginary component
    return np.sqrt(np.power(data,2) + np.power(hilb,2))


def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided


def smooth(x, window_len=11, window='hanning'):
    """
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
          flat window will produce a moving average smoothing.

    output:
       the smoothed signal

    """
    # if x.ndim != 1:
    #     raise ValueError("smooth only accepts 1 dimension arrays.")
    # if x.size < window_len:
    #     raise ValueError("Input vector needs to be bigger than window size.")
    # if window_len<3:
    #     return x
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # s=np.r_[2*x[0]-x[int(window_len-1)::-1], x, 2*x[-1]-x[-1:int(-window_len):-1]]

    a = (2*x[:,0]-x[:,int(window_len-1)::-1].T).T
    b = (2*x[:,-1]-x[:,-1:int(-window_len):-1].T).T
    s = np.hstack([a, x, b])

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    # y=np.convolve(w/w.sum(),s,mode='same')
    return np.apply_along_axis(np.convolve, 1, s, w/w.sum(), mode='same')[:, int(window_len):int(-window_len+1)]


def rms(tr):
    '''
    Root mean square
    '''
    return np.sqrt(np.sum(tr**2, axis=1)/float(len(tr)))


window_size = 5

Datalist = ['EQ','Tremor','Noise']
for name in Datalist:
    #地震資料檔案路徑
    f_mseed = glob.glob(os.path.join(f'/content/drive/My Drive/2025wintercamp_EQlab/Data/{name}/KMNB*pkl'))
    f_mseed.sort()
    f_mseed

    globals()[f'{name}_trace_sw'] = []

    with open(f_mseed[0], 'rb') as f:
        score = pickle.load(f)

    if name == 'Noise':
    ii = 1500
    else:
    ii = 63

    for i in range(0,ii):
    sps = score[i][0].stats['sampling_rate']
    window_p = int(window_size * sps)
    trace_sw = sliding_window(score[i][0].data, window_p, stepsize=window_p)
    trace_sw = trace_sw.astype(float)
    trace_sw_cleaned = trace_sw[~np.isnan(trace_sw).any(axis=1)]
    globals()[f'{name}_trace_sw'].append(trace_sw_cleaned)

    globals()[f'{name}_trace_sw'] = np.concatenate(globals()[f'{name}_trace_sw'], axis=0)

# 初始化空的特徵陣列，用來存放所有資料的特徵
Feature_array = []

for name in Datalist:
    for i in range(len(globals()[f'{name}_trace_sw'][:, 0])):
        # 提取特徵並將其轉換為列表
        Feature_temp = list(Feature_Engineering(globals()[f'{name}_trace_sw'][i, :]))

        # 根據資料來源設定對應的標籤
        if name == 'EQ':
            label = 0  # 地震 (EQ) 標籤為 0
        elif name == 'Tremor':
            label = 1  # 顫抖 (Tremor) 標籤為 1
        else:
            label = 2  # 噪音 (Noise) 標籤為 2

        # 將標籤和對應的特徵一起放入特徵陣列
        Feature_array.append([label] + Feature_temp)

# 將特徵陣列轉換為 NumPy 陣列
Feature_array = np.array(Feature_array)

# 創建 DataFrame，特徵名稱為 'feature_1', 'feature_2', ..., 'feature_n'
df_F = pd.DataFrame(Feature_array[:, 1:], columns=[f'feature_{i+1}' for i in range(len(Feature_array[0][1:]))])

# 為 DataFrame 加入 'labels' 欄位
df_F['labels'] = Feature_array[:, 0]

# 調整欄位順序，將 'labels' 放在最前面
df_F = df_F[['labels'] + [f'feature_{i+1}' for i in range(len(Feature_array[0][1:]))]]

# 顯示結果 DataFrame
df_F


#區分資料集
# 刪除 'labels' 欄位，將其存放在 X (特徵資料集)
X = df_F.drop(columns=['labels'])
# 將 'labels' 欄位存放在 y (目標資料集)
y = df_F['labels']

# 使用 train_test_split 將資料集分為訓練集和測試集，測試集佔20%，並設置隨機種子為42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 顯示訓練集中的各類標籤數量
print(f'Train Dataset Label0 : {len(y_train[y_train == 0])}')
print(f'Train Dataset Label1 : {len(y_train[y_train == 1])}')
print(f'Train Dataset Label2 : {len(y_train[y_train == 2])}')
print(f'Test Dataset Label0 : {len(y_test[y_test == 0])}')
print(f'Test Dataset Label1 : {len(y_test[y_test == 1])}')
print(f'Test Dataset Label2 : {len(y_test[y_test == 2])}')


# 標準化處理
# 生成特徵名稱的列表，例如 ['feature_1', 'feature_2', ..., 'feature_n']
Feature_list = [f'feature_{i+1}' for i in range(len(Feature_array[0][1:]))]

# 對每個特徵進行標準化 (Standardization)
for Fname in Feature_list:
    # 計算訓練集中該特徵的平均值
    F_mean = X_train[Fname].mean()
    # 計算訓練集中該特徵的標準差
    F_std = X_train[Fname].std()
    # 使用訓練集的平均值和標準差對訓練集進行標準化
    X_train[Fname] = (X_train[Fname] - F_mean) / F_std
    # 使用訓練集的平均值和標準差對測試集進行標準化
    X_test[Fname] = (X_test[Fname] - F_mean) / F_std

# 檢查 NaN 值（訓練集部分）
# 判斷每個元素是否為 NaN，結果是一個布林值 DataFrame
nan_values = X_train.isna()
# 計算每一列中的 NaN 數量
nan_count = nan_values.sum()
# 選擇那些包含至少一個 NaN 值的欄位
non_zero_nan_columns = nan_count[nan_count > 0]
print(f"訓練集中每個特徵的 NaN 值數量:\n{non_zero_nan_columns}")
# 計算訓練集中總的 NaN 值數量
total_nan = nan_values.sum().sum()
print(f"訓練集中的 NaN 值總數: {total_nan}")

# 檢查 NaN 值（測試集部分）
# 判斷每個元素是否為 NaN
nan_values = X_test.isna()  # 或者可以使用 X_test.isnull()
# 計算每一列中的 NaN 數量
nan_count = nan_values.sum()
# 選擇那些包含至少一個 NaN 值的欄位
non_zero_nan_columns = nan_count[nan_count > 0]
print(f"測試集中每個特徵的 NaN 值數量:\n{non_zero_nan_columns}")
# 計算測試集中總的 NaN 值數量
total_nan = nan_values.sum().sum()
print(f"測試集中的 NaN 值總數: {total_nan}")

# 刪除 有 NaN 值的部分
X_train = X_train.drop(columns=['feature_13'])
X_test = X_test.drop(columns=['feature_13'])

# 再次檢查 NaN 值
nan_values = X_train.isna()
nan_count = nan_values.sum()
non_zero_nan_columns = nan_count[nan_count > 0]
print(f"Train NaN values count per feature:\n{non_zero_nan_columns}")
total_nan = nan_values.sum().sum()
print(f"Total number of NaN values in X_train: {total_nan}")

nan_values = X_test.isna()  # 或者 X_train.isnull()
nan_count = nan_values.sum()  # 每一列中的 NaN 數量
non_zero_nan_columns = nan_count[nan_count > 0]
print(f"Test NaN values count per feature:\n{non_zero_nan_columns}")
total_nan = nan_values.sum().sum()
print(f"Total number of NaN values in X_test: {total_nan}")

#K-近鄰演算法（KNN）實踐地震分類
def KNN_model(X_train_normalized, X_test_normalized, y_train, y_test, feature_names ,n_neighbors=5):
    # 建立 KNN 分類器
    # n_neighbors: K 值，表示選擇最近鄰的數量
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(X_train_normalized, y_train)  # 使用訓練數據進行模型訓練

    # 預測訓練和測試數據的結果
    y_pred = classifier.predict(X_test_normalized)       # 測試數據的預測結果
    y_train_pred = classifier.predict(X_train_normalized)  # 訓練數據的預測結果

    # 計算評估指標
    Train_acc = accuracy_score(y_train, y_train_pred)  # 訓練數據的準確率
    Test_acc = accuracy_score(y_test, y_pred)          # 測試數據的準確率
    precision = precision_score(y_test, y_pred, average='weighted')  # 加權平均精確率
    recall = recall_score(y_test, y_pred, average='weighted')        # 加權平均召回率
    f1 = f1_score(y_test, y_pred, average='weighted')                # 加權平均 F1 分數

    # 計算並繪製混淆矩陣（訓練數據）
    cm = confusion_matrix(y_train, y_train_pred)  # 混淆矩陣（訓練數據）
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("KNN Confusion Matrix Train")  # 圖表標題
    plt.xlabel("Predicted")                   # X 軸標籤：預測類別
    plt.ylabel("Actual")                      # Y 軸標籤：實際類別
    plt.show()

    # 計算並繪製混淆矩陣（測試數據）
    cm = confusion_matrix(y_test, y_pred)  # 混淆矩陣（測試數據）
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("KNN Confusion Matrix Test")  # 圖表標題
    plt.xlabel("Predicted")                  # X 軸標籤：預測類別
    plt.ylabel("Actual")                     # Y 軸標籤：實際類別
    plt.show()

    # 輸出評估指標結果
    print(f"KNN Train Accuracy: {Train_acc}")  # 訓練數據的準確率
    print(f"KNN Test Accuracy: {Test_acc}")    # 測試數據的準確率
    print(f"KNN Precision: {precision}")       # 精確率
    print(f"KNN Recall: {recall}")             # 召回率
    print(f"KNN F1 Score: {f1}")               # F1 分數

    # 返回評估指標和預測結果
    return Train_acc, Test_acc, precision, recall, f1, y_train_pred, y_pred

#模型運作
KNN_Train_acc, KNN_Test_acc, KNN_precision, KNN_recall, KNN_f1, KNN_y_train_pred, KNN_y_pred = KNN_model(X_train, X_test, y_train, y_test,Feature_list)


k_range = range(1, 51 ,2)  # 設定 K 值範圍，這裡從 1 到 50，間隔為 2
train_accuracies = []  # 儲存每個 K 值下的訓練準確率
test_accuracies = []   # 儲存每個 K 值下的測試準確率

for k in k_range:
    # 建立 KNN 分類器
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)  # 使用訓練數據進行模型訓練

    # 預測
    y_train_pred = classifier.predict(X_train)  # 預測訓練數據的結果
    y_test_pred = classifier.predict(X_test)    # 預測測試數據的結果

    # 計算準確率
    train_acc = accuracy_score(y_train, y_train_pred)  # 訓練數據的準確率
    test_acc = accuracy_score(y_test, y_test_pred)    # 測試數據的準確率

    # 存儲結果
    train_accuracies.append(train_acc)  # 將訓練準確率加入列表
    test_accuracies.append(test_acc)    # 將測試準確率加入列表

# 找出最佳 K 值
best_k = k_range[np.argmax(test_accuracies)]  # 測試準確率最高時的 K 值
print(f"最佳 K 值為: {best_k}")  # 輸出最佳 K 值
print(f"測試準確率為: {max(test_accuracies)}")  # 輸出最佳測試準確率

# 繪製 K 值與準確率的關係圖
plt.figure(figsize=(10, 6))
plt.plot(k_range, train_accuracies, label="Train Accuracy", marker='o')  # 繪製訓練準確率曲線
plt.plot(k_range, test_accuracies, label="Test Accuracy", marker='o')    # 繪製測試準確率曲線
plt.xlabel("K Value")  # X 軸標籤
plt.ylabel("Accuracy")  # Y 軸標籤
plt.title("K Value & Accuracy")  # 圖表標題
plt.legend()  # 顯示圖例
plt.grid()  # 顯示網格線
plt.show()

"""# **調整後的新模型**"""

#K-近鄰演算法（KNN）實踐地震分類
def KNN_model(X_train_normalized, X_test_normalized, y_train, y_test, feature_names ,n_neighbors=5,model_filename='KNN_model.joblib'):
    # 建立分類器
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(X_train_normalized, y_train)

    # 預測
    y_pred = classifier.predict(X_test_normalized)
    y_train_pred = classifier.predict(X_train_normalized)

    # 計算評估指標
    Train_acc = accuracy_score(y_train, y_train_pred)
    Test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 混淆矩陣
    cm = confusion_matrix(y_train, y_train_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("New KNN Confusion Matrix Train")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("New KNN Confusion Matrix Test")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(f"New KNN Train Accuracy: {Train_acc}")
    print(f"New KNN Test Accuracy: {Test_acc}")
    print(f"New KNN Precision: {precision}")
    print(f"New KNN Recall: {recall}")
    print(f"New KNN F1 Score: {f1}")

    joblib.dump(classifier, model_filename)# 將訓練好的模型輸出
    print(f"模型已經保存到 {model_filename}")

    return Train_acc, Test_acc, precision, recall, f1, y_train_pred, y_pred

KNN_Train_acc, KNN_Test_acc, KNN_precision, KNN_recall, KNN_f1, KNN_y_train_pred, KNN_y_pred = KNN_model(X_train, X_test, y_train, y_test,Feature_list,n_neighbors=best_k)


def DT_model(X_train_normalized, X_test_normalized, y_train, y_test, feature_names, max_depth=15, min_samples_leaf=1, min_samples_split=5):
    # 建立分類器（決策樹模型）
    classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=42)
    classifier.fit(X_train_normalized, y_train)  # 訓練模型

    # 預測（測試集和訓練集）
    y_pred = classifier.predict(X_test_normalized)  # 測試集的預測結果
    y_train_pred = classifier.predict(X_train_normalized)  # 訓練集的預測結果

    # 繪製決策樹
    plt.figure(figsize=(12, 8))  # 設定圖表大小
    plot_tree(classifier, feature_names=feature_names, filled=True, rounded=True)  # 繪製決策樹
    plt.show()

    # 計算評估指標
    Train_acc = accuracy_score(y_train, y_train_pred)  # 訓練集的準確率
    Test_acc = accuracy_score(y_test, y_pred)  # 測試集的準確率
    precision = precision_score(y_test, y_pred, average='weighted')  # 加權精確率
    recall = recall_score(y_test, y_pred, average='weighted')  # 加權召回率
    f1 = f1_score(y_test, y_pred, average='weighted')  # 加權 F1 分數

    # 畫出混淆矩陣（訓練集）
    cm = confusion_matrix(y_train, y_train_pred)  # 訓練集的混淆矩陣
    plt.figure(figsize=(8, 6))  # 設定圖表大小
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))  # 熱圖
    plt.title("DT Confusion Matrix Train")  # 標題
    plt.xlabel("Predicted")  # X 軸標籤（預測值）
    plt.ylabel("Actual")  # Y 軸標籤（實際值）
    plt.show()

    # 畫出混淆矩陣（測試集）
    cm = confusion_matrix(y_test, y_pred)  # 測試集的混淆矩陣
    plt.figure(figsize=(8, 6))  # 設定圖表大小
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))  # 熱圖
    plt.title("DT Confusion Matrix Test")  # 標題
    plt.xlabel("Predicted")  # X 軸標籤（預測值）
    plt.ylabel("Actual")  # Y 軸標籤（實際值）
    plt.show()

    # 印出評估指標
    print(f"DT Train Accuracy: {Train_acc}")  # 訓練集準確率
    print(f"DT Test Accuracy: {Test_acc}")  # 測試集準確率
    print(f"DT Precision: {precision}")  # 精確率
    print(f"DT Recall: {recall}")  # 召回率
    print(f"DT F1 Score: {f1}")  # F1 分數

    # 返回評估指標和預測結果
    return Train_acc, Test_acc, precision, recall, f1, y_train_pred, y_pred

#模型運作
DT_Train_acc, DT_Test_acc, DT_precision, DT_recall, DT_f1, DT_y_train_pred, DT_y_pred = DT_model(X_train, X_test, y_train, y_test,Feature_list)

"""# **模型的調整**"""

# 定義參數網格
param_grid = {
    'max_depth': [5, 10, 15, 20, 25, None],  # None 表示不限制深度
    'min_samples_leaf': [1, 2, 5, 10],  # 每個葉節點的最小樣本數
    'min_samples_split': [2, 5, 10, 20]  # 執行分裂所需的最小樣本數
}

# 建立分類器（決策樹模型）
dt = DecisionTreeClassifier(random_state=42)  # 設定隨機種子以確保結果可重現

# 使用 GridSearchCV 進行超參數調整
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)  # cv 表示交叉驗證的摺數
grid_search.fit(X_train, y_train)  # 在訓練集上執行網格搜索

# 獲得最佳參數
best_params = grid_search.best_params_  # 最佳參數組合
best_score = grid_search.best_score_  # 最佳交叉驗證準確率

print(f"最佳參數: {best_params}")  # 印出最佳參數
print(f"交叉驗證準確率: {best_score}")  # 印出最佳交叉驗證準確率

# 使用最佳參數建立最終模型
best_dt = grid_search.best_estimator_  # 獲取最佳模型

# 評估測試集準確率
test_acc = best_dt.score(X_test, y_test)  # 測試集上的準確率
print(f"測試集準確率: {test_acc}")  # 印出測試集準確率

"""# **調整後的新模型**"""

#決策樹（DT）實踐地震分類
def DT_model(X_train_normalized, X_test_normalized, y_train, y_test, feature_names ,max_depth=15, min_samples_leaf=1, min_samples_split=5,model_filename='DT_model.joblib'):
    # 建立分類器
    classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=42)
    classifier.fit(X_train_normalized, y_train)

    # 預測
    y_pred = classifier.predict(X_test_normalized)
    y_train_pred = classifier.predict(X_train_normalized)

    # 繪製決策樹
    plt.figure(figsize=(12, 8))
    plot_tree(classifier, feature_names=Feature_list, filled=True, rounded=True)
    plt.show()

    # 計算評估指標
    Train_acc = accuracy_score(y_train, y_train_pred)
    Test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 混淆矩陣
    cm = confusion_matrix(y_train, y_train_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("New DT Confusion Matrix Train")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("New DT Confusion Matrix Test")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(f"New DT Train Accuracy: {Train_acc}")
    print(f"New DT Test Accuracy: {Test_acc}")
    print(f"New DT Precision: {precision}")
    print(f"New DT Recall: {recall}")
    print(f"New DT F1 Score: {f1}")

    joblib.dump(classifier, model_filename)# 將訓練好的模型輸出
    print(f"模型已經保存到 {model_filename}")

    return Train_acc, Test_acc, precision, recall, f1, y_train_pred, y_pred
DT_Train_acc, DT_Test_acc, DT_precision, DT_recall, DT_f1, DT_y_train_pred, DT_y_pred = DT_model(X_train, X_test, y_train, y_test,Feature_list,max_depth=10, min_samples_leaf=10, min_samples_split=2)

# 支援向量機（SVM）實踐地震分類
def SVM_model(X_train_normalized, X_test_normalized, y_train, y_test, feature_names, kernel='linear', C=1, gamma='auto'):
    # 建立分類器
    if kernel == 'linear':
        # 當使用線性核函數時，不需要 gamma 參數
        classifier = SVC(kernel=kernel, C=C, random_state=42)
    else:
        # 當使用非線性核函數時，需指定 gamma 參數
        classifier = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)

    # 訓練模型
    classifier.fit(X_train_normalized, y_train)

    # 預測
    y_pred = classifier.predict(X_test_normalized)  # 測試集的預測結果
    y_train_pred = classifier.predict(X_train_normalized)  # 訓練集的預測結果

    # 計算評估指標
    Train_acc = accuracy_score(y_train, y_train_pred)  # 訓練集準確率
    Test_acc = accuracy_score(y_test, y_pred)  # 測試集準確率
    precision = precision_score(y_test, y_pred, average='weighted')  # 加權平均精確率
    recall = recall_score(y_test, y_pred, average='weighted')  # 加權平均召回率
    f1 = f1_score(y_test, y_pred, average='weighted')  # 加權平均 F1 分數

    # 繪製訓練集的混淆矩陣
    cm = confusion_matrix(y_train, y_train_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("SVM Confusion Matrix Train")
    plt.xlabel("Predicted")  # 預測值
    plt.ylabel("Actual")  # 實際值
    plt.show()

    # 繪製測試集的混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("SVM Confusion Matrix Test")
    plt.xlabel("Predicted")  # 預測值
    plt.ylabel("Actual")  # 實際值
    plt.show()

    # 打印評估結果
    print(f"SVM 訓練集準確率: {Train_acc}")
    print(f"SVM 測試集準確率: {Test_acc}")
    print(f"SVM 精確率: {precision}")
    print(f"SVM 召回率: {recall}")
    print(f"SVM F1 分數: {f1}")

    # 返回評估指標與預測結果
    return Train_acc, Test_acc, precision, recall, f1, y_train_pred, y_pred

#模型運作
SVM_Train_acc, SVM_Test_acc, SVM_precision, SVM_recall, SVM_f1, SVM_y_train_pred, SVM_y_pred = SVM_model(X_train, X_test, y_train, y_test,Feature_list)


# 定義參數網格
param_grid = {
    'C': [0.1, 1, 10, 100],  # 正則化參數
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # 核函數
    'gamma': ['scale', 'auto']  # 核函數係數
}

# 建立 SVM 分類器
svm = SVC(random_state=42)

# 使用 GridSearchCV
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# 獲得最佳參數
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"最佳參數: {best_params}")
print(f"交叉驗證準確率: {best_score}")

# 使用最佳參數建立最終模型
best_svm = grid_search.best_estimator_

# 評估測試集準確率
test_acc = best_svm.score(X_test, y_test)
print(f"測試集準確率: {test_acc}")


#支援向量機（SVM）實踐地震分類
def SVM_model(X_train_normalized, X_test_normalized, y_train, y_test, feature_names ,kernel='linear', C=1 ,gamma='auto', model_filename="svm_model.joblib"):
    # 建立分類器
    if kernel=='linear':
      classifier = SVC(kernel=kernel, C=C, random_state=42)
    else:
      classifier = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    classifier.fit(X_train_normalized, y_train)

    # 預測
    y_pred = classifier.predict(X_test_normalized)
    y_train_pred = classifier.predict(X_train_normalized)

    # 計算評估指標
    Train_acc = accuracy_score(y_train, y_train_pred)
    Test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 混淆矩陣

    cm = confusion_matrix(y_train, y_train_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("SVM Confusion Matrix Train")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("SVM Confusion Matrix Test")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(f"SVM Train Accuracy: {Train_acc}")
    print(f"SVM Test Accuracy: {Test_acc}")
    print(f"SVM Precision: {precision}")
    print(f"SVM Recall: {recall}")
    print(f"SVM F1 Score: {f1}")

    joblib.dump(classifier, model_filename)# 將訓練好的模型輸出
    print(f"模型已經保存到 {model_filename}")


    return Train_acc, Test_acc, precision, recall, f1, y_train_pred, y_pred

SVM_Train_acc, SVM_Test_acc, SVM_precision, SVM_recall, SVM_f1, SVM_y_train_pred, SVM_y_pred = SVM_model(X_train, X_test, y_train, y_test,Feature_list,kernel='linear', C=100)


#補充集成學習實踐地震分類

def ensemble_model(X_train, X_test, y_train, y_test,model_filename='voting_model.joblib'):
    # 建立個別的基分類器
    svm_clf = SVC(kernel='linear', C=100, probability=True, random_state=42)
    dt_clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, min_samples_split=2, random_state=42)
    knn_clf = KNeighborsClassifier(n_neighbors=13)

    # 建立投票分類器（可以選擇硬投票或軟投票）
    voting_clf = VotingClassifier(
        estimators=[
            ('svm', svm_clf),
            ('dt', dt_clf),
            ('knn', knn_clf)
        ],
        voting='soft'  # 'hard' 為硬投票，'soft' 為軟投票
    )

    # 訓練集成模型
    voting_clf.fit(X_train, y_train)

    # 預測
    y_train_pred = voting_clf.predict(X_train)
    y_test_pred = voting_clf.predict(X_test)

    # 計算評估指標
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    # 混淆矩陣
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("Ensemble Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 結果輸出
    print(f"Ensemble Train Accuracy: {train_acc}")
    print(f"Ensemble Test Accuracy: {test_acc}")
    print(f"Ensemble Precision: {precision}")
    print(f"Ensemble Recall: {recall}")
    print(f"Ensemble F1 Score: {f1}")

    joblib.dump(voting_clf, model_filename)
    print(f"模型已經保存到 {model_filename}")

    return voting_clf, train_acc, test_acc, precision, recall, f1

voting_model, train_acc, test_acc, precision, recall, f1 = ensemble_model(X_train, X_test, y_train, y_test)
