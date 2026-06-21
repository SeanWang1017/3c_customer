"""
启动 FastAPI 客服工作台
- 默认地址：http://localhost:8089
- 用法：从项目根目录跑 `python src/web/run.py`
"""
import sys
from pathlib import Path

# 加项目根目录到 sys.path，让 uvicorn 能找到 src 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.web.app:app",
        host="0.0.0.0",
        port=8089,
        reload=True,
        log_level="info",
    )
