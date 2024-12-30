#!/bin/bash
if [ "$1" = "musl" ];then
export RTT_CC_PREFIX=riscv64-unknown-linux-musl-
export RTT_EXEC_PATH=/media/psf/workspace/sophgo/host-tools/gcc/riscv64-linux-musl-x86_64/bin
else
export RTT_CC_PREFIX=riscv64-unknown-elf-
export RTT_EXEC_PATH=/media/psf/workspace/sophgo/host-tools/gcc/riscv64-elf-x86_64/bin
fi
