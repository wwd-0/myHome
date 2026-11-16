#!/usr/bin/env bash
# ==============================================================================
# MyHome Zigbee 门锁固件 Docker 一键构建脚本 (无需本地安装 Zephyr/VS Code)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output_firmware"
mkdir -p "${OUTPUT_DIR}"

IMAGE="nordicsemi/nrfconnect-sdk:v2.6.0"

echo "========================================================================"
echo "🚀 启动 Docker 容器编译 Pro Micro nRF52840 (Nice!Nano v2) Zigbee 固件..."
echo "   镜像: ${IMAGE}"
echo "========================================================================"

docker run --rm \
  -v "${SCRIPT_DIR}:/workspace/app" \
  -v "${OUTPUT_DIR}:/workspace/output" \
  -w /workspace \
  "${IMAGE}" \
  bash -c "
    set -e
    west init -m https://github.com/nrfconnect/sdk-nrf --mr v2.6.0 /workspace/ncs
    cd /workspace/ncs
    west update --narrow -o=--depth=1
    cd /workspace/app
    west build -b nice_nano_v2 -- -DCONFIG_BUILD_OUTPUT_UF2=y
    cp -v build/zephyr/zephyr.uf2 /workspace/output/
    cp -v build/zephyr/zephyr.hex /workspace/output/
    echo '✅ 编译完成！UF2 固件已输出至: /workspace/output/zephyr.uf2'
  "

echo ""
echo "🎉 构建成功！固件文件已保存在："
ls -lh "${OUTPUT_DIR}/zephyr.uf2"
echo "👉 双击开发板 RST 引脚弹出 U 盘，将 zephyr.uf2 拖入即可完成烧录！"

