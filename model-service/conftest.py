"""pytest 配置。

把 `model-service/` 放进 sys.path，测试里就能直接 `from app.embedder import ...`。
不这么做的话，从仓库根跑 `pytest model-service/tests` 会 ImportError。
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
