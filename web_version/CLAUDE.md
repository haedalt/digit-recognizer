# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this folder.

## Overview

손글씨 숫자 인식기의 **웹 버전**. 브라우저의 캔버스에 마우스/터치로 숫자를 그리면 MNIST로 학습한 모델이 숫자를 예측하고 숫자별 확률 막대를 보여준다. 기능은 데스크톱 버전(`../desktop_version/`)과 같게 유지한다.

서버 없이 **브라우저에서 예측까지 모두 실행하는 정적 사이트**이며, GitHub Pages로 호스팅한다. Python은 모델 학습/내보내기에만 쓴다.

## Structure

- `index.html`, `style.css`, `script.js` — 280x280 `<canvas>`, Predict/Clear 버튼, 예측 결과와 확률 막대. `script.js`는 시작 시 `model.json`/`model.bin`을 `fetch`로 불러온다.
- `model.js` — Python 코드의 JavaScript 이식: `preprocess()`(Pillow LANCZOS 리사이즈 포함)와 MLP 순전파(`predictProba`). 브라우저와 Node 양쪽에서 동작하도록 DOM에 의존하지 않는다.
- `export_model.py` — `mnist_mlp.joblib`를 불러오거나(없으면 MNIST를 내려받아 학습) 가중치를 `model.bin`(float32 리틀엔디언; W1, b1, W2, b2, ... 순서, W는 (입력, 출력) 행 우선)과 `model.json`(층 크기)으로 내보낸다.
- `model.bin`, `model.json` — 배포되는 모델. git에 커밋한다. (`*.joblib`은 `.gitignore`로 제외)

## Commands

- 로컬 실행: 이 폴더에서 `python -m http.server 8000` 후 `http://127.0.0.1:8000` 접속. `fetch`를 쓰므로 `index.html`을 파일로 직접 열면 동작하지 않는다.
- 모델 갱신: `pip install numpy pillow scikit-learn joblib` 후 `python export_model.py`. 재학습하려면 먼저 `mnist_mlp.joblib` 삭제 (MNIST 다운로드에 네트워크 필요, 학습 수 분 소요).
- 배포: `main` 브랜치에 push하면 `../.github/workflows/pages.yml`이 이 폴더를 GitHub Pages에 올린다.

## Architecture guidelines

- **전처리는 Python과 동일해야 함:** `model.js`의 `preprocess()`는 데스크톱 버전과 같은 단계(바운딩 박스 자르기 → 긴 변 20px, Python식 반올림 → 28x28 중앙 배치 → 무게중심 이동(Pillow AFFINE 최근접 샘플링) → [0, 1] 스케일)를 따른다. 리사이즈는 Pillow의 고정소수점 LANCZOS를 그대로 재현해 결과가 거의 일치한다(±1 밝기). 전처리를 바꾸면 Python 결과와 비교해 확인한다.
- **MNIST 규약 유지:** 캔버스는 검은 바탕에 흰 글씨로 그린다. Clear 시 검은색으로 다시 채운다. 회색조 값은 캔버스 픽셀의 R 채널을 쓴다.
- **클래스 순서:** `fetch_openml` 레이블은 문자열 `'0'..'9'`이므로 `export_model.py`가 출력층 열을 숫자 순서로 재정렬해 저장한다. JS는 출력 인덱스 = 숫자로 가정한다.
- **네트워크 구조:** JS는 ReLU 은닉층 + softmax 출력만 구현한다. `MLPClassifier`의 activation이나 구조를 바꾸면 `predictProba`도 함께 바꾼다.
- 마우스/터치를 뗄 때 자동으로 예측한다(Pointer Events 사용).

## Notes

- 데스크톱 버전과 전처리를 공유하므로, 한쪽을 바꾸면 다른 쪽도 함께 맞춘다.
