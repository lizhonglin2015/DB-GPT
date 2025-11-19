#!/bin/bash

# DB-GPT 启动脚本
# 功能：停止现有进程并启动新的 DB-GPT 服务

set -e

# 配置参数
CONFIG_FILE="${1:-configs/dbgpt-proxy-tongyi.toml}"
LOG_DIR="${LOG_DIR:-.}"

# 生成日志文件名（格式：dbgpt.MM-DD.HH.log）
LOG_FILE="${LOG_DIR}/dbgpt.$(date +%m-%d.%H).log"

echo "=========================================="
echo "DB-GPT 启动脚本"
echo "=========================================="
echo "配置文件: ${CONFIG_FILE}"
echo "日志文件: ${LOG_FILE}"
echo ""

# 步骤1: 停止现有进程
echo "步骤1: 检查并停止现有 DB-GPT 进程..."
EXISTING_PIDS=$(pgrep -f dbgpt || true)

if [ -n "$EXISTING_PIDS" ]; then
    echo "发现运行中的进程: $EXISTING_PIDS"
    echo "正在停止..."
    pgrep -f dbgpt | xargs kill -9
    sleep 2
    echo "已停止现有进程"
else
    echo "未发现运行中的 DB-GPT 进程"
fi

echo ""

# 步骤2: 启动新服务
echo "步骤2: 启动 DB-GPT 服务..."
echo "使用命令: uv run dbgpt start webserver --config ${CONFIG_FILE}"
echo "日志输出到: ${LOG_FILE}"
echo ""

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_FILE")"

# 启动服务
nohup uv run dbgpt start webserver --config "${CONFIG_FILE}" > "${LOG_FILE}" 2>&1 &

# 获取新启动的进程ID
NEW_PID=$!
sleep 2

# 验证进程是否成功启动
if ps -p $NEW_PID > /dev/null 2>&1; then
    echo "✓ DB-GPT 服务已成功启动"
    echo "  进程ID: $NEW_PID"
    echo "  日志文件: ${LOG_FILE}"
    echo ""
    echo "查看日志: tail -f ${LOG_FILE}"
    echo "停止服务: kill $NEW_PID"
else
    echo "✗ 服务启动可能失败，请检查日志: ${LOG_FILE}"
    exit 1
fi

echo "=========================================="

