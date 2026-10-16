#!/usr/bin/env bash
# ==============================================================================
# MyHome 智能门禁控制系统 - 一键启动脚本
# ==============================================================================

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 [1/3] 正在编译/检查 C++ 服务..."
cmake -S cpp_service -B cpp_service/build > /dev/null
make -C cpp_service/build -j8 > /dev/null

echo "🛑 [2/3] 清理旧端口占用..."
pkill -f "door_person_detector/service.py" 2>/dev/null || true
pkill -f "myhome_verify_server" 2>/dev/null || true
sleep 1

echo "🔥 [3/3] 启动统一服务集群..."
# 1. 启动 PyTorch MPS 人体感知微服务 (端口 8092)
/Users/a58/miniforge3/envs/comfyui/bin/python door_person_detector/service.py --port 8092 > /dev/null 2>&1 &

# 等待微服务就绪
for i in {1..20}; do
  if curl -s http://127.0.0.1:8092/health > /dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

# 2. 启动 C++ HTTP 网关与硬件联动协调服务 (端口 8090)
./cpp_service/build/myhome_verify_server > /dev/null 2>&1 &

echo ""
echo "========================================================================"
echo "✅ 服务已全部在后台启动就绪！"
echo "========================================================================"
echo "🌐 控制台地址:   http://127.0.0.1:8090/"
echo "⚡ 人体感知API:   http://127.0.0.1:8092/health"
echo "🚪 认证模式:     [方式1: 人体+手势] [方式2: 人体+智能开关] [方式3: 声音(预留)]"
echo "🛑 停止服务命令: ./stop_service.sh"
echo "========================================================================"
