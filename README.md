## Transformer 이해를 돕기위한 튜토리얼 제작

> <a href='https://yangoos57.github.io/blog/DeepLearning/paper/Transformer/Transformer_From_Scratch/'>도식화로 논문 이해하기 : Transformer
> </a> 포스트에 대한 Repo입니다.

### 프로젝트 소개

- Pytorch 활용해 Transformer를 구현하고, 이를 학습하고 평가하는 방법에 대해 소개함.
- Multi-30k 데이터를 활용해 :fr: 프랑스어 -> :us: 영어 번역을 수행하는 Transformer 제작

<br/>

### 이런 경우 활용하면 좋습니다.

- Transformer를 Pytorch로 구현하는 방법에 대해 알고싶은 경우
- 모델 구현은 했으나 모델 내부에서 데이터가 흐르는 과정에 대한 이해가 어려운 경우
- 직접 구현한 Transfomrer를 활용해 학습, 평가를 수행하고 싶은 경우

<br/>

### 세부 내용

- 학습 및 테스트 과정에 대한 이해를 돕기 위해 `[tutorial] training transformer.ipynb`와 `[tutorial] test transformer.ipynb` 작성

<br/>

- 모델의 학습 과정을 실시간으로 확인할 수 있도록 helper 함수 구현

  ```python
  1번째 epoch 실행
  ------------------------------

  Dataset is "training"

  200번째 batch에 있는 0번째 문장 예측 결과 확인
  src :  Un homme en uniforme orange pose au milieu d' une rue .
  prd :  A man in a suit shirt is in front air of the building . <eos> . . . . . . . <eos> . . <eos> . . . <eos>
  trg :  A man in an orange uniform poses in the middle of a street .

  Dataset is "validation"

  5번째 batch에 있는 0번째 문장 예측 결과 확인
  src :  Un vieil homme est assis avec un plateau sur ses genoux .
  prd :  A man man is on a red in a field . <eos> . . . . . . . . . . . . . <eos> <eos>
  trg :  An old man sits with a tray in his lap .

  ----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*
  Epoch: 1, Train loss: 3.586, Val loss: 3.639, Epoch time = 33.573s
  ----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*

  2번째 epoch 실행
  ------------------------------

  Dataset is "training"

  200번째 batch에 있는 0번째 문장 예측 결과 확인
  src :  Un homme en uniforme orange pose au milieu d' une rue .
  prd :  A man in a suit dress is in front street of a building . <eos> . . . . . . . . . . . . . . .
  trg :  A man in an orange uniform poses in the middle of a street .

  Dataset is "validation"

  5번째 batch에 있는 0번째 문장 예측 결과 확인
  src :  Un vieil homme est assis avec un plateau sur ses genoux .
  prd :  A man man is on a bicycle on a field . <eos> . . . . . . . . . . . . . . <eos>
  trg :  An old man sits with a tray in his lap .

  ----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*
  Epoch: 2, Train loss: 3.496, Val loss: 3.577, Epoch time = 33.021s
  ----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*

  3번째 epoch 실행
  ------------------------------

  Dataset is "training"

  200번째 batch에 있는 0번째 문장 예측 결과 확인
  src :  Un homme en uniforme orange pose au milieu d' une rue .
  prd :  A man in a suit is are in a water of the crowd . <eos> . . . . . . . . . . . . . . .
  trg :  A man in an orange uniform poses in the middle of a street .

  Dataset is "validation"

  5번째 batch에 있는 0번째 문장 예측 결과 확인
  src :  Un vieil homme est assis avec un plateau sur ses genoux .
  prd :  A older man is on a red on a mouth . <eos> . . . . . . . . . . <eos> . . . <eos>
  trg :  An old man sits with a tray in his lap .

  ----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*
  Epoch: 3, Train loss: 3.417, Val loss: 3.535, Epoch time = 33.396s
  ----*----*----*----*----*----*----*----*----*----*----*----*----*----*----*
  ```

<br/>

- 모델에서 활용하는 Argument가 어떤 용도로, 어떻게 활용되는지 이해를 도울 수 있는 주석 제공

  ```python
  # 예시
  class EncoderBlock(nn.Module):
      def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
          """
          config 참고

          embed_size(=512) : embedding 차원
          heads(=8) : Attention 개수
          dropout(=0.1): Node 학습 비율
          forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                  forward_expension * embed_size(2*512 = 1024)
          """
  ```

<br/>

- 모델 내부에서 발생하는 데이터 흐름 및 Tensor의 차원 변경을 추적할 수 있는 주석 제공

  ```python
    # 예시
      def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src)
        # (n,1,src_token_len,src_token_len) 4

        trg_mask = self.make_trg_mask(trg)
        # (n,1,trg_token_len,trg_token_len) 4

        src_trg_mask = self.make_pad_mask(trg, src)
        # (n,1,trg_token_len,src_token_len) 4
  ```

 <br/>

### 구동환경

```
torch == 1.12.1
torchtext == 0.13.1
torchdata == 0.4.1
spacy == 3.4.3
pandas == 1.4.3
```

<br/>

### 참고자료

참고자료를 함께 활용해 Transformer 구현 및 학습에 대해 다각도로 이해하는 것을 추천

<a href='https://github.com/Huffon/Pytorch-transformer-kor-eng'>devfon님 구현자료 (한,영 번역 및 sub-word embedding) </a>

<a href='https://github.com/cpm0722/transformer_Pytorch'>Hansu Kim님 구현자료
(이론에 대한 상세한 설명 및 이를 기반으로 한 모델 구현) </a>

<a href='https://github.com/hyunwoongko/transformer'>Kevin Ko님 구현자료(구조에 대한 간결한 설명 및 이해하기 쉬운 코드 구조)</a>

<a href='https://tutorials.Pytorch.kr/beginner/translation_transformer.html'>Pytorch 공식 튜토리얼 </a>
