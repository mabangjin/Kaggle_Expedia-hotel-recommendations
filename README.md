# Kaggle_Expedia-hotel-recommendations

https://www.kaggle.com/c/expedia-hotel-recommendations

## Abstract
Planning your dream vacation, or even a weekend escape, can be an overwhelming affair. With hundreds, even thousands, of hotels to choose from at every destination, it's difficult to know which will suit your personal preferences. Should you go with an old standby with those pillow mints you like, or risk a new hotel with a trendy pool bar? 

![icon](https://kaggle2.blob.core.windows.net/competitions/kaggle/5056/media/expedia_icons.png)


Expedia wants to take the proverbial rabbit hole out of hotel search by providing personalized hotel recommendations to their users. This is no small task for a site with hundreds of millions of visitors every month!

Currently, Expedia uses search parameters to adjust their hotel recommendations, but there aren't enough customer specific data to personalize them for each user. In this competition, Expedia is challenging Kagglers to contextualize customer data and predict the likelihood a user will stay at 100 different hotel groups.

The data in this competition is a random selection from Expedia and is not representative of the overall statistics.

## Evaluation
Submissions are evaluated according to the [Mean Average Precision @ 5](https://www.kaggle.com/wiki/MeanAveragePrecision) (MAP@5):


where |U| is the number of user events, P(k) is the precision at cutoff k, n is the number of predicted hotel clusters.


## Daily Diary

### 2017.2.3  (John Park님 피드백 포함)

- Expedia Hotel Recommendation 시작
- 데이터의 용량 약 3.79g -> pandas에서 읽을시 약 5g
- row : 37,670,293  / Feature : 24
- 2013, 2014년 데이터를 기반으로 2015년의 호텔 추천( y : hotel_cluster / MAP@5 )
- 2013 : 2014 = 1:2의 비율

- 데이터를 볼 때, 시각적으로 그래프를 그리는 것도 좋지만 그냥 데이터를 직접 들여다보는 방법을 추천(pd.set_option("max_columns", 40) 활용)
- 처음엔 데이터를 3~5메가정도로 줄여보고 모델링을 시작하는 것이 좋다
- 위에서 5000~10000~100000명 정도로 잘라보기 => 단, 위에서 자르는 방법도 있지만 아래와 같이 랜덤으로 추출을 여러번 해서 검증
~~~
    df1 = df.ix[np.random.choice(df.index, 10000)]
~~~
- 샘플링을 통해 랜덤포레스트의 important_feature가 변하는지 파악 => 만약 변한다면 데이터가 흔들린다는 증거 => 이런 경우 해결책은?
- is_mobile같이 0, 1인 데이터는 재미가 없다. 복잡하게 나와있는 데이터가 더 의미가 있다는 것
- check_in, out 데이터는 가로로 볼 것 => 요일같은 경우는 평일 / 주말 / 시즌별로 볼 것
- y 값은 보통 제일 좌측 혹은 우측에 두는데, 좌측에 두고 데이터를 움직여보면 더 편하다


#### 할 일
- 데이터 랜덤 추출 후, 랜덤포레스트 결과값 달라지는지 파악 (row, column 둘 다 실행)
- [h2o](http://h2o.ai) 설치 및 공부 ( 머신러닝/딥러닝을 쉽게 돌릴 수 있음 => 한글 자료가 적으니 추후 번역해보면 좋을 것 같다 )
- Feature Engineering 하기 전 Train set 더 완성하기
- 데이터 바라보기


