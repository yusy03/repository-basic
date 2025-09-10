```
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```
cd /content/drive/MyDrive/
```

    /content/drive/MyDrive


# 매트플롯립 한글 폰트 다운
"부록3 매트플롯립 입문"에서 한글 폰트를 올바르게 출력하기 위한 설치 방법을 설명했다. 설치 방법은 다음과 같다.


```
!sudo apt-get install -y fonts-nanum* | tail -n 1
!sudo fc-cache -fv
!rm -rf ~/.cache/matplotlib
```

    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78, <> line 4.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Processing triggers for fontconfig (2.13.1-4.2ubuntu5) ...
    /usr/share/fonts: caching, new cache contents: 0 fonts, 1 dirs
    /usr/share/fonts/truetype: caching, new cache contents: 0 fonts, 3 dirs
    /usr/share/fonts/truetype/humor-sans: caching, new cache contents: 1 fonts, 0 dirs
    /usr/share/fonts/truetype/liberation: caching, new cache contents: 16 fonts, 0 dirs
    /usr/share/fonts/truetype/nanum: caching, new cache contents: 39 fonts, 0 dirs
    /usr/local/share/fonts: caching, new cache contents: 0 fonts, 0 dirs
    /root/.local/share/fonts: skipping, no such directory
    /root/.fonts: skipping, no such directory
    /usr/share/fonts/truetype: skipping, looped directory detected
    /usr/share/fonts/truetype/humor-sans: skipping, looped directory detected
    /usr/share/fonts/truetype/liberation: skipping, looped directory detected
    /usr/share/fonts/truetype/nanum: skipping, looped directory detected
    /var/cache/fontconfig: cleaning cache directory
    /root/.cache/fontconfig: not cleaning non-existent cache directory
    /root/.fontconfig: not cleaning non-existent cache directory
    fc-cache: succeeded


* 모든 설치가 끝나면 한글 폰트를 바르게 출력하기 위해 **[런타임]** -> **[세션 다시시작]** 을 클릭한 다음, 아래 셀부터 코드를 실행해 주십시오.


```
# 라이브러리 임포트

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# 폰트 관련 용도
import matplotlib.font_manager as fm

# 나눔 고딕 폰트의 경로 명시
path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_name = fm.FontProperties(fname=path, size=10).get_name()
```


```
# 파이토치 관련 라이브러리
import torch
```


```
# 기본 폰트 설정
plt.rcParams['font.family'] = font_name

# 기본 폰트 사이즈 변경
plt.rcParams['font.size'] = 14

# 기본 그래프 사이즈 변경
plt.rcParams['figure.figsize'] = (6,6)

# 기본 그리드 표시
# 필요에 따라 설정할 때는, plt.grid()
plt.rcParams['axes.grid'] = True

# 마이너스 기호 정상 출력
plt.rcParams['axes.unicode_minus'] = False

# 넘파이 부동소수점 자릿수 표시
np.set_printoptions(suppress=True, precision=4)
```


```
# [문제 1] 코드
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 1. 데이터 준비
wine = load_wine()
# 13개의 특징 중 2개만 사용 (알코올, 색상 강도)
X = wine.data[:, [0, 9]]
# 3개의 품종 중 2개만 사용 (class 0, 1)
y = wine.target[wine.target != 2]
X = X[wine.target != 2]

# 2. 연습 문제와 실전 시험 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 학습 (연습 문제로만 학습!)
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. 모델 평가 (실전 시험으로 평가!)
accuracy = model.score(X_test, y_test)
print(f"모델의 실전 시험 정확도: {accuracy * 100:.2f}%")

# 5. 결과 시각화 (코드는 참고용)
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=60, alpha=0.8)
plt.title("와인 품종 분류 (실전 시험 결과)")
plt.xlabel("알코올 도수")
plt.ylabel("색상 강도")
plt.show()
```

    모델의 실전 시험 정확도: 100.00%



    
![png](INT_ML_EX1-2_Week2_files/INT_ML_EX1-2_Week2_8_1.png)
    


### **1주차 실습 과제**

**목표**: 지도/비지도 학습, Train/Test 데이터 분리의 개념을 복습하고, 선형 모델의 비용(Cost) 개념을 코드로 직접 확인합니다.

#### **문제 1: 지도 학습 - 와인 품종 분류**

와인의 두 가지 화학 성분(feature)으로 와인 품종(class)을 분류하는 문제입니다. `____` 부분을 채워, 모델을 학습시키고 처음 보는 시험 문제(Test set)에 대한 정확도를 확인하세요.


```
# [문제 1] 코드
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 1. 데이터 준비
wine = load_wine()
# 13개의 특징 중 2개만 사용 (알코올, 색상 강도)
X = wine.data[:, [0, 9]]
# 3개의 품종 중 2개만 사용 (class 0, 1)
y = wine.target[wine.target != 2]
X = X[wine.target != 2]

# 2. 연습 문제와 실전 시험 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 모델 학습 (연습 문제로만 학습!)
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. 모델 평가 (실전 시험으로 평가!)
accuracy = model.score(X_test, y_test)
print(f"모델의 실전 시험 정확도: {accuracy * 100:.2f}%")

# 5. 결과 시각화 (코드는 참고용)
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=60, alpha=0.8)
plt.title("와인 품종 분류 (실전 시험 결과)")
plt.xlabel("알코올 도수")
plt.ylabel("색상 강도")
plt.show()
```

    모델의 실전 시험 정확도: 94.87%



    
![png](INT_ML_EX1-2_Week2_files/INT_ML_EX1-2_Week2_10_1.png)
    


#### **문제 2: 비지도 학습 - 고객 그룹 찾기**

한 쇼핑몰의 고객 데이터(연수입, 소비 점수)가 있습니다. 이 데이터를 보고 숨겨진 고객 그룹(Cluster)을 찾아내는 문제입니다. `____` 부분을 채워, K-Means 모델로 4개의 고객 그룹을 찾아내고 시각화하세요.


```
# [문제 2] 코드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. 가상 고객 데이터 준비 (정답 없음)
X_customers = np.array([
    [25, 78], [30, 85], [22, 95], # 그룹1
    [80, 21], [88, 15], [95, 25], # 그룹2
    [28, 25], [35, 18], [33, 22], # 그룹3
    [85, 91], [78, 85], [91, 94]  # 그룹4
])

# 2. K-Means 모델로 학습 (데이터 전체를 보고 구조를 파악)
# 총 4개의 그룹을 찾으려고 합니다.
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_customers)

# 3. 각 데이터가 어느 그룹에 속하는지 결과 확인
predicted_labels = kmeans.labels_

# 4. 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_customers[:, 0], X_customers[:, 1], c=predicted_labels, cmap='viridis', s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.title("고객 그룹 군집화 결과")
plt.xlabel("연수입 (Annual Income)")
plt.ylabel("소비 점수 (Spending Score)")
plt.show()
```


    
![png](INT_ML_EX1-2_Week2_files/INT_ML_EX1-2_Week2_12_0.png)
    


#### **문제 3: 선형 회귀 - 모델 성능 비교**

아이스크림 가게의 하루 평균 기온(feature)과 아이스크림 판매량(target) 데이터가 있습니다. 두 개의 후보 모델(직선) 중 어떤 모델이 처음 보는 시험 문제(Test set)를 더 잘 예측하는지 Cost(비용)를 계산하여 비교하세요.


```
# [문제 3] 코드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 데이터 준비 (기온과 판매량)
temperature = np.array([[20], [22], [25], [28], [30], [31], [33], [35]])
sales = np.array([150, 180, 240, 300, 340, 360, 400, 430])
X_train, X_test, y_train, y_test = train_test_split(temperature, sales, test_size=0.4, random_state=1)

# 2. 후보 모델 정의
# 모델 1: W=15, b=-120
# 모델 2: W=20, b=-220
W1, b1 = 15, -120
W2, b2 = 20, -220

# 3. '실전 시험' 데이터로 각 모델의 예측값 계산
pred1_test = W1 * X_test + b1
pred2_test = W2 * X_test + b2

# 4. '실전 시험' 데이터로 각 모델의 비용(MSE) 계산
cost1_test = np.mean((pred1_test - y_test)**2)
cost2_test = np.mean((pred2_test - y_test)**2)

# 5. 결과 시각화 및 비교
plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, color='gray', alpha=0.5, label='Train Data')
plt.scatter(X_test, y_test, color='blue', label='Test Data', s=100)
plt.plot(temperature, W1 * temperature + b1, 'r--', label=f'Model 1 (Test Cost: {cost1_test:.2f})')
plt.plot(temperature, W2 * temperature + b2, 'g:', label=f'Model 2 (Test Cost: {cost2_test:.2f})')
plt.title("어떤 모델이 판매량을 더 잘 예측할까?")
plt.xlabel("평균 기온 (Temperature)")
plt.ylabel("아이스크림 판매량 (Sales)")
plt.legend()
plt.show()

# 최종 결론 출력
if cost1_test < cost2_test:
    print("결론: 모델 1이 실전 문제를 더 잘 예측합니다.")
else:
    print("결론: 모델 2가 실전 문제를 더 잘 예측합니다.")
```


    
![png](INT_ML_EX1-2_Week2_files/INT_ML_EX1-2_Week2_14_0.png)
    


    결론: 모델 1이 실전 문제를 더 잘 예측합니다.

