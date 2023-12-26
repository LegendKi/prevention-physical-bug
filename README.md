# PPB(prevention-physical-bug)
openCV와 openpose api 기반 어깨 비대칭여부를 실시간으로 분류 해주는 프로젝트

## 🖥️ 프로젝트 소개
PPB(Prevention Physical Bug) 프로젝트는 체계적인 물리적 불균형의 예방을 목표로 합니다. OpenPose API를 활용하여 이미지 데이터에서 머리, 목, 그리고 양쪽 어깨의 관절 좌표를 정밀하게 추출하고 추출된 데이터는 후속 처리 과정을 거친 후 최적화된 학습 모델에 적용됩니다. OpenCV 라이브러리를 통해 실시간 웹캠 영상을 분석하며, 이를 통해 얻어진 관절 좌표는 OpenPose API를 통해 처리됩니다. 학습된 모델을 사용하여 어깨의 비대칭 여부를 실시간으로 판단하고, 사용자에게 화면을 통해 즉시 피드백을 제공하는 시스템을 구현하였습니다. 이 프로젝트는 사람들의 바르지못한 자세를 조기에 감지하고 이를 예방하는 데 중점을 두는 것이 목표입니다.


## 개발 기간
23.09 - 23.10

### 🧰 개발환경
- python 3.9.12
- OpenCV 4.8.1
- OpenPose