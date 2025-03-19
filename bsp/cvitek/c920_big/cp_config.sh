#!/bin/bash

# 拷贝配置脚本
# 用法: ./copy-config.sh [0-7]

# 检查参数是否存在
if [ $# -ne 1 ]; then
    echo "错误：需要1个参数（0-7）"
    echo "用法: \$0 [0-7]"
    exit 1
fi

# 获取输入数字
NUM=$1

# 检查数字有效性
if ! [[ $NUM =~ ^[0-7]$ ]]; then
    echo "错误：参数必须为 0-7 的数字"
    exit 2
fi

# 定义目标目录和文件名模板
DEST_DIR="configs"
CONFIG_FILE=".config"
HEADER_FILE="rtconfig.h"
TEMPLATE="tp_Image_${NUM}"

# 检查源文件存在性
if [ ! -f $CONFIG_FILE ] || [ ! -f $HEADER_FILE ]; then
    echo "错误：源文件缺失！"
    [ ! -f $CONFIG_FILE ] && echo "缺少: $CONFIG_FILE"
    [ ! -f $HEADER_FILE ] && echo "缺少: $HEADER_FILE"
    exit 3
fi

# 创建目标目录（如果不存在）
mkdir -p $DEST_DIR

# 执行拷贝操作
echo "正在拷贝配置 #$NUM..."
cp -v $CONFIG_FILE "${DEST_DIR}/${TEMPLATE}.config"
cp -v $HEADER_FILE "${DEST_DIR}/${TEMPLATE}.h"

echo "操作完成！"
