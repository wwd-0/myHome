#!/usr/bin/env bash
# ==============================================================================
# MyHome 智能门禁控制系统 - 一键停止脚本
# ==============================================================================

echo "🛑 正在停止门禁相关服务进程..."
pkill -f "door_person_detector/service.py" 2>/dev/null || true
pkill -f "myhome_verify_server" 2>/dev/null || true
sleep 1
echo "✅ 所有门禁后台服务已停止。"
