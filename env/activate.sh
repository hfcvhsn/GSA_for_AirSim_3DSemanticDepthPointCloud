# Source after activating rgbd-sem3d. Paths can be overridden in resources.json.
_SEM3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${SEM3D_CUDA_HOME:-$(python -c "import sys;sys.path.insert(0,sys.argv[1]);from src.config import load_config;print(load_config()['paths']['cuda_home'])" "$_SEM3D_ROOT")}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="$_SEM3D_ROOT/.cache/huggingface"
export TORCH_HOME="$_SEM3D_ROOT/.cache/torch"
export MPLCONFIGDIR="$_SEM3D_ROOT/.cache/matplotlib"
export PIP_CACHE_DIR="$_SEM3D_ROOT/.cache/pip"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset _SEM3D_ROOT
