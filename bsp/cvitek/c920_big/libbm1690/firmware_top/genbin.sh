#!/bin/bash
BIN_NAME="c920.bin"

dd if=c920_0.bin of=$BIN_NAME
# c920_1_offset = CONFIG_FW_SIZE = 0X2000000 = 32M
dd if=c920_1.bin of=$BIN_NAME bs=1M seek=32
