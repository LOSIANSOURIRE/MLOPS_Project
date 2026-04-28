# # # # # syntax=docker/dockerfile:1
# # # # FROM python:3.10-slim

# # # # # 1. Install uv
# # # # COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# # # # # 2. System settings for cache performance
# # # # # UV_LINK_MODE=copy is required when using cache mounts
# # # # ENV UV_LINK_MODE=copy \
# # # #     UV_COMPILE_BYTECODE=1

# # # # WORKDIR /app

# # # # # 3. Install curl
# # # # RUN apt-get update && apt-get install -y --no-install-recommends \
# # # #     curl && rm -rf /var/lib/apt/lists/*

# # # # # 4. Copy requirements
# # # # COPY requirements.txt .

# # # # # 5. CACHED INSTALLATION
# # # # # This mount stays on your host machine across different builds.
# # # # RUN --mount=type=cache,target=/root/.cache/uv \
# # # #     uv pip install --system \
# # # #     --find-links https://download.pytorch.org/whl/cpu \
# # # #     torch torchvision -r requirements.txt

# # # # # 6. Verification
# # # # RUN python3 -c "import torch; print('Using Torch:', torch.__version__)"

# # # # COPY . .
# # # # CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# # # # syntax=docker/dockerfile:1
# # # FROM python:3.10-slim

# # # # 1. Install uv
# # # COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# # # # 2. Optimization settings
# # # ENV UV_LINK_MODE=copy \
# # #     UV_COMPILE_BYTECODE=1 \
# # #     PYTHONUNBUFFERED=1

# # # WORKDIR /app

# # # # 3. System deps
# # # RUN apt-get update && apt-get install -y --no-install-recommends \
# # #     curl && rm -rf /var/lib/apt/lists/*

# # # # 4. Copy requirements
# # # COPY requirements.txt .

# # # # 5. THE FIX: Explicitly request the +cpu versions
# # # # We use --index-url for Torch and --extra-index-url for everything else
# # # RUN --mount=type=cache,target=/root/.cache/uv \
# # #     uv pip install --system \
# # #     --index-url https://download.pytorch.org/whl/cpu \
# # #     --extra-index-url https://pypi.org/simple \
# # #     torch==2.4.0+cpu torchvision==0.19.0+cpu -r requirements.txt

# # # # 6. Verify (The build will fail here if it's wrong)
# # # RUN python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# # # COPY . .
# # # CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# # # syntax=docker/dockerfile:1
# # FROM python:3.10-slim

# # # 1. Install uv
# # COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# # # 2. Optimization settings
# # ENV UV_LINK_MODE=copy \
# #     UV_COMPILE_BYTECODE=1 \
# #     PYTHONUNBUFFERED=1

# # WORKDIR /app

# # # 3. System deps (Apt-get)
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     curl && rm -rf /var/lib/apt/lists/*

# # # 4. Copy requirements
# # COPY requirements.txt .

# # # 5. THE FIX: Added --index-strategy unsafe-best-match
# # # This allows uv to combine the CPU index and the Standard index safely
# # RUN --mount=type=cache,target=/root/.cache/uv \
# #     uv pip install --system \
# #     --index-url https://download.pytorch.org/whl/cpu \
# #     --extra-index-url https://pypi.org/simple \
# #     --index-strategy unsafe-best-match \
# #     torch==2.4.0+cpu torchvision==0.19.0+cpu -r requirements.txt

# # # 6. Verification
# # RUN python3 -c "import torch; print('Using Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# # COPY . .
# # CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# # syntax=docker/dockerfile:1
# FROM python:3.10-slim

# # 1. Install uv
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# # 2. System dependencies (Required for Cartopy, Rasterio, and OpenCV)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl \
#     libgdal-dev libgeos-dev libproj-dev binutils g++ \
#     libgl1 libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# ENV UV_LINK_MODE=copy \
#     UV_COMPILE_BYTECODE=1 \
#     PYTHONUNBUFFERED=1

# WORKDIR /app

# # 3. Copy requirements
# COPY requirements.txt .

# # 4. THE CPU INSTALLATION
# # - We force torch/torchvision to the CPU index
# # - We swap tensorflow for tensorflow-cpu to save 2GB
# # - We use --index-strategy unsafe-best-match to resolve the mix
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv pip install --system \
#     --index-url https://download.pytorch.org/whl/cpu \
#     --extra-index-url https://pypi.org/simple \
#     --index-strategy unsafe-best-match \
#     torch==2.4.0+cpu \
#     torchvision==0.19.0+cpu \
#     tensorflow-cpu==2.17.0 \
#     -r requirements.txt

# # 5. Verification
# RUN python3 -c "import torch; import tensorflow as tf; print('Torch CPU:', torch.__version__); print('TF CPU:', tf.__version__)"

# COPY . .
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# syntax=docker/dockerfile:1


FROM python:3.10-slim

# 1. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. System dependencies (Needed for Cartopy, Rasterio, and OpenCV)
# This step is slow (15 mins), but Docker will cache it after the first run.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    libgdal-dev libgeos-dev libproj-dev binutils g++ \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Optimization settings
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# 4. Copy requirements
COPY requirements.txt .

# 5. THE INSTALLATION
# Fixed the 'o' in opencv-python-headless
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    torch==2.4.0+cpu \
    torchvision==0.19.0+cpu \
    tensorflow-cpu==2.17.0 \
    -r requirements.txt

# 6. Verification
RUN python3 -c "import torch; import tensorflow as tf; print('Success!')"

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]