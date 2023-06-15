#!/bin/bash

set -euo pipefail

git clone --depth 1 https://github.com/antoinelame/GazeTracking.git
cd GazeTracking
rm -rf .git

cat <<EOF > pyproject.toml
[tool.poetry]
name = "gaze_tracking"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.8"
numpy = "^1.22.0"
opencv-python = "^4.2.0.32"
dlib = "^19.16.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
EOF

poetry install
poetry build

cd -
mkdir -p deps/
mv GazeTracking/dist/*.whl deps/
rm -rf GazeTracking/
