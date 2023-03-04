# RememberAI
## Image Clustering with Encoded Space (Try)


## Develop Diary
### 개발 플랜

1. 같은 이미지를 볼 때 분위기, 상황을 분류해서 학습 → 분위기 학습기랑 event Classifier를 따로 학습시켜서 이 둘의 결과를 합치는 걸로!
    1. 데이터셋에 학습을 시키긴 해야할 거 같다. 해당 데이터셋들에 pretrained된 건 많지 않아보임
2. 자동 질문 생성기 → 이거 진짜 좀 시간 걸릴듯… 뭔가 끝나고나서라도 만들고 싶다. 일단 1번 다 개발하고 그 다음에 시작할듯
3. 여차하면 커뮤니티 악성 게시글 탐지기도 만들 수 있음 근데 우선순위에선 뭐로 둬야할지는 아직…!

→ 쓸만 한 Class(Wider Dataset)

- 0 Parade, 7 Cheering, 12 Group, 18 Concert, 19 Couple, 20 Family_Group, 21 Festival, 22 Picnic, 28 Sports_Fan, 29 Students_Schoolkids, 34 Baseball, 35 Basketball, 37 Soccer, 38 Tennis, 39 Ice_Skating, 41 Swimming, 49 Greeting, 50 Celebration_Or_Party
- 일단 이것들만 가지고 학습?


### Issues

- WIDER Dataset으로 학습을 시켜봤는데 그리 성능이 좋지 못함… 논문을 좀 몇 개 읽어보니 Event 를 Classification 하는 거 자체가 애초에 높은 정확도를 얻기가 힘든듯.. 흠…
    - 원인 : Class 불균형 + 너무 지저분한 데이터.
        - 딱 봐도 CNN 모델이 파악하기 어려운 이미지들이 많음. 그냥 저스틴 비버가 농구장에 있다는 이유로 농구라고 분류한 경우도 있었음. 이런 이미지들이 너무너무 많았음.
        - 데이터셋 자체가 깔끔하지도 않고 불균형도 심했다. 데이터를 한번 쫙 정리하고 학습을 돌려보면 달라질 수 있음
        
    - 대처
        1. Class Weight 조정 1/len(classes) 로 조정 → 효과 x
        2. 모델 변경 → 큰 효과 x (아직까진 Inception V3 = Mobile V2)
        3. Dataset 변경 → USED라는 데이터셋이 있는데 일단 해볼 예정
    
**2월 6일**

일단 전처리까지는 성공했는데, 문제는 용량이 너무 커서 Colab 32기가로도 못 버팀 ㅋㅋ 

→ 일단 쓸데없는 Class 삭제하고, Under Sampling으로 최대한 데이터 수 줄여나가면서 진행

if USED로도 그렇게 성능이 좋지 못하면… 

→ WIDER를 활용한 평균 정확도가 40퍼 였음. 여기서 Class 다 균형 맞추고 데이터 정제 쭉 하고 해야 할듯. 

→ 시간과 노가다가 필요하지만… 성능을 위해서라면 어쩔 수 없다이….

⇒ 그래도 이 방법은 최후의 수단으로…!
    
**2월 8일**
    
USED 데이터 언더 샘플링코드까진 다 짰음 → 이제 코랩이 감당할 수 있을 만큼언더샘플링의 수를 지정하는 게 좋을 것같다.


**2월 10일**

USED 데이터셋이 깔끔한 편임에도 불구하고 사실상 거의 찍는 수준에 불과한 결과가 나왔다. 왜 그럴까. 데이터셋을 전달하는 방식이 잘못된 건지… 어떤 건지… shuffle을 하고 넣으면 좀 달라지지 않을까 싶긴 하지만, 기본적으로 성능 자체가 학습을 안 하다 싶이 하는 꼴이라 다른 방법도 고려해봐야 할 것같다. 생각난 방법은 다음과 같다.

1. 각 이벤트마다 “대표 이미지”를 선정해서 들어온 이미지에 대해 그 대표 이미지들과 유사한 곳으로 매핑하는 방법
    
    ***HOW***
    
    - 봤을 때, 해당 이벤트를 대표적으로 나타내는 이미지 10개 선정 → 이들을 모두 projection 시킴(by Feature Map을 차원축소)
        
        → 대표 이미지를 어떻게 선정할 것인가? → 군집화를 해서 가장 중심에 있는 10개의 이미지? 아니면 가장자리에 있는 10개의 이미지? 
        
        → 일단 중심이 좋을 것같은 게, 중심일 수록 해당 Event의 특징을 대표한다고 생각해볼 수는 있을 것같음… 가장자리는 다른 class와 헷갈릴 위험이 존재. (애매모한 특징을 대표적인 특징이다!라고 할 가능성 있음)
        
    - 예를 들어, 졸업 사진 10개의 특징을 추출해서 이 10개의 종합적인 특징을 어떤 공간에 표현하는 것 이런 식으로 졸업사진, 운동사진 등등의 이미지들의 종합 특징을 space에 표현하면 독립된 군집으로 나타날 것으로 기대(왜냐면 확실하지가 않음)
    - 이렇게 나타낸 상태에서 query이미지가 들어옴(군집으로 표현해도 됨, 어차피 앨범작업이니까.. 하나씩 하면 시간이 너무 오래 걸릴 거 같긴 함)
    - query image가 들어오면 이걸 DB이미지와 똑같이 처리를 한 다음 그걸 Class DB가 있는 Space에 표현
    - space상에 표현된 것들의 거리를 다 잼→ 유클리안 or 코사인 (뭐가 더 좋은지는 실험으로 알아보자)
    - 즉, 위의 과정을 한 마디로 요약하면, 기존의 Event들 중 가장 유사한 걸 찾아주는 방식
    - 한번 해보자고,..,,!!

# 최종 전략
- 추억이 없는 경우 -> Image Captioning 후 Tokenizer(Eng Tokenizer) -> Bert Embedding (SBERT, HuggingFace) -> Vectorized words Clustering
- 추억이 있는 경우 -> Just Tokenizer -> Bert Embedding -> Vectorized words Clustering