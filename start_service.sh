#!/usr/bin/env bash
# ==============================================================================
# MyHome 智能门禁控制系统 - 一键启动脚本 (纯 C++ 原生推理，无需 Python 服务)
# ==============================================================================

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

unset http_proxy https_proxy ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="*"
export NO_PROXY="*"

if [ ! -f "cpp_service/build/myhome_verify_server" ]; then
    echo "🚀 [1/3] 正在编译 C++ 服务..."
    cmake -S cpp_service -B cpp_service/build > /dev/null
    make -C cpp_service/build -j8 > /dev/null
else
    echo "🚀 [1/3] C++ 服务二进制已就绪，跳过编译..."
fi

echo "🛑 [2/3] 清理旧端口占用..."
pkill -f "door_person_detector/service.py" 2>/dev/null || true
pkill -f "myhome_verify_server" 2>/dev/null || true
sleep 1

echo "🔥 [3/3] 启动纯 C++ 智能门禁控制服务 (端口 8090)..."
./cpp_service/build/myhome_verify_server > /dev/null 2>&1 &

# 等待服务就绪
for i in {1..20}; do
  if curl -s http://127.0.0.1:8090/health > /dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

echo ""
echo "========================================================================"
echo "✅ 纯 C++ 原生门禁系统已启动就绪 (0 Python 运行时依赖)！"
echo "========================================================================"
echo "🌐 控制台地址:   http://127.0.0.1:8090/"
echo "⚡ 原生感知API:   http://127.0.0.1:8090/health"
echo "🚪 认证模式:     [方式1: 人体+手势] [方式2: 人体+智能开关] [方式3: 声音(预留)]"
echo "🛑 停止服务命令: ./stop_service.sh"
echo "========================================================================"
