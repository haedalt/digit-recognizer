# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this folder.

## Overview

손글씨 숫자 인식기의 **데스크톱 버전**. 단일 파일 Tkinter 앱(`digit_recognizer.py`)에서 마우스로 숫자를 그리면, MNIST로 학습한 scikit-learn `MLPClassifier`가 숫자를 예측하고 숫자별 확률 막대를 보여준다.

현재 원본 코드(`digit_recognizer.py`, `Run Digit Recognizer.bat`, `mnist_mlp.joblib`)는 상위 폴더(`Study-01/`)에 있다. 이 폴더로 옮긴 뒤에는 아래 설명이 그대로 적용된다. `MODEL_PATH`와 `.bat`의 경로는 스크립트 위치 기준(`__file__`, `%~dp0`)이므로 함께 옮기면 수정할 필요가 없다.

## Commands

- 콘솔 출력과 함께 실행 (학습 진행 상황, 오류 확인 가능): `python digit_recognizer.py`
- 콘솔 창 없이 실행: `Run Digit Recognizer.bat` (`pythonw` 사용 → traceback이 보이지 않으므로 디버깅 시에는 위 명령 사용)
- 의존성 (requirements 파일 없음): `pip install numpy pillow scikit-learn joblib`. Tkinter는 표준 Python 설치에 포함.
- 재학습: `mnist_mlp.joblib` 삭제. 다음 실행 시 `fetch_openml`로 MNIST를 내려받고(네트워크 필요) 학습한다(수 분 소요).

테스트, 린터 설정, 빌드 단계는 없다.

## Architecture

- **모델 캐싱:** `load_or_train_model()`은 스크립트 폴더의 `mnist_mlp.joblib`가 있으면 불러오고, 없으면 `mnist_784`를 내려받아 표준 60k/10k 분할로 학습 후 저장한다. 캐시 파일은 pickle된 sklearn 객체라 scikit-learn 버전이 바뀌면 로드가 실패할 수 있다. 그 경우 재학습한다.
- **이중 그리기 표면:** 모든 획은 화면의 `tk.Canvas`와 오프스크린 PIL `Image`(`self.image`, 280x280, 모드 `L`)에 두 번 그려진다. 예측은 PIL 이미지만 읽으므로, 그리기/지우기 로직을 추가하거나 바꿀 때는 두 곳을 모두 갱신해야 한다.
- **전처리는 MNIST와 일치해야 함:** `preprocess()`는 바운딩 박스로 자르고, 긴 변을 20px로 축소하고, 28x28 안에 배치한 뒤 무게중심을 이미지 중앙으로 옮기고, 픽셀을 [0, 1]로 스케일한다. 검은 바탕에 흰 글씨(MNIST 규약)여야 한다. 이 부분의 변경은 정확도에 직접 영향을 준다.
- **클래스 레이블은 문자열:** `fetch_openml`의 레이블은 `'0'..'9'`이므로 `predict()`는 `model.classes_`를 이용해 `predict_proba` 출력을 숫자 순서로 재정렬한다. 학습 데이터 출처를 바꾸더라도 이 매핑은 유지해야 한다.
- 예측은 Predict 버튼뿐 아니라 마우스를 뗄 때도 자동으로 실행된다.

## Notes

- 웹 버전은 `../web_version/`에 별도로 있다. 전처리 로직을 바꾸면 웹 버전과 동작이 달라지지 않도록 함께 확인한다.
- git 저장소 루트는 상위 폴더 `Study-01/`이다. `*.joblib`은 `.gitignore`로 제외되므로 새로 clone하면 첫 실행 때 학습한다.
