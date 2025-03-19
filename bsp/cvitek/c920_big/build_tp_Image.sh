#!/bin/bash

# 配置恢复与编译脚本
# 用法: ./restore-build.sh [0-8]

# 定义目标目录和文件模板
DEST_DIR="configs"
CONFIG_FILE=".config"
HEADER_FILE="rtconfig.h"

# --------------- 函数定义 ---------------
do_compile() {
    local num=$1
    local config_src="${DEST_DIR}/tp_Image_${num}.config"
    local header_src="${DEST_DIR}/tp_Image_${num}.h"

    # 检查源文件是否存在
    if [[ ! -f $config_src || ! -f $header_src ]]; then
        echo -e "\033[31m错误：配置文件 ${num} 不存在！\033[0m"
        [ ! -f $config_src ] && echo "缺失文件: $config_src"
        [ ! -f $header_src ] && echo "缺失文件: $header_src"
        return 1
    fi

    # 覆盖文件
    echo -e "\n\033[34m===== 正在应用配置 #${num} =====\033[0m"
    cp -v "$config_src" "$CONFIG_FILE"
    cp -v "$header_src" "$HEADER_FILE"

    # 清理并编译
    echo -e "\n\033[33m执行编译任务...\033[0m"
    scons -c && scons -j2
    local build_status=$?
    
    if [ $build_status -ne 0 ]; then
        echo -e "\033[31m编译配置 #${num} 失败！\033[0m"
        return $build_status
    else
        echo -e "\033[32m配置 #${num} 编译成功！\033[0m"
    fi
}

# --------------- 主逻辑 ---------------
# 参数检查
if [ $# -ne 1 ] || ! [[ $1 =~ ^[0-8]$ ]]; then
    echo "错误：参数必须为 0-8 的数字"
    echo "用法: \$0 [0-8]"
    echo "  输入 0-7 : 编译特定配置"
    echo "  输入 8   : 批量编译所有配置(0-7)"
    exit 1
fi

input_num=$1

if [ $input_num -eq 8 ]; then
    # 批量模式
    for num in {0..7}; do
        do_compile $num || break  # 任一失败则终止
    done
else
    # 单配置模式
    do_compile $input_num
fi

rm -rf *.elf
scp tp_Image_* test@172.26.14.164://home/test/sheep/kenneth/
