# LoomCV

A computer vision tool for digital artists learning perspective and figure drawing. Upload a reference photo and get automatic perspective analysis, a Loomis-style 3D mannequin overlay, and real-time directional shading — all interactive in the browser.

Named after Andrew Loomis, whose figure drawing and perspective books remain foundational references for artists today.

---

## How It Works

1. Upload a reference photo
2. The C++ engine detects perspective vanishing points (Hough + RANSAC) and human pose joints (MoveNet via ONNX Runtime)
3. The React frontend overlays a perspective grid, 3D Loomis mannequin(s), and a draggable light indicator
4. Place perspective-correct boxes in the scene; export the annotated image as PNG

---

## Stack

| Layer | Tech |
| --- | --- |
| Vision engine | C++17, OpenCV 4, Eigen 3, ONNX Runtime, cpp-httplib, nlohmann/json |
| Frontend | React 18, TypeScript, Three.js, Zustand, TanStack Query, Tailwind CSS |
| Deploy | Docker, Fly.io (engine), Vercel (frontend), GitHub Actions |

---

## Local Development

### Engine

```bash
cmake -S engine -B build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Debug
./build/Debug/engine.exe <image_path>
```

### Frontend *(coming Week 3)*

```bash
cd web && npm install && npm run dev
```

---
