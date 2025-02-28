import os

# toolchains options
ARCH        ='risc-v'
VENDOR      ='t-head'
CPU         ='c920'
CROSS_TOOL  ='gcc'

if os.getenv('RTT_ROOT'):
    RTT_ROOT = os.getenv('RTT_ROOT')
else:
    RTT_ROOT = r'../../..'

if os.getenv('RTT_CC'):
    CROSS_TOOL = os.getenv('RTT_CC')

if  CROSS_TOOL == 'gcc':
    PLATFORM    = 'gcc'
    EXEC_PATH   = r'/opt/Xuantie-900-gcc-elf-newlib-x86_64-V2.8.1/bin'
else:
    print('Please make sure your toolchains is GNU GCC!')
    exit(0)

if os.getenv('RTT_EXEC_PATH'):
    EXEC_PATH = os.getenv('RTT_EXEC_PATH')

BUILD = 'debug'
CHIP_TYPE = 'sg2260'

if PLATFORM == 'gcc':
    # toolchains
    #PREFIX  = 'riscv64-unknown-elf-'
    PREFIX  = os.getenv('RTT_CC_PREFIX') or 'riscv64-unknown-elf-'
    CC      = PREFIX + 'gcc'
    CXX     = PREFIX + 'g++'
    AS      = PREFIX + 'gcc'
    AR      = PREFIX + 'ar'
    LINK    = PREFIX + 'gcc'
    TARGET_EXT = 'elf'
    SIZE    = PREFIX + 'size'
    OBJDUMP = PREFIX + 'objdump'
    OBJCPY  = PREFIX + 'objcopy'

    DEVICE  = ' -mcmodel=medany -march=rv64imafdcv '
    CFLAGS  = DEVICE + ' -Wno-cpp -fvar-tracking -ffreestanding -fno-common -ffunction-sections -fdata-sections -fstrict-volatile-bitfields -D_POSIX_SOURCE '
    CFLAGS += '-D__PTHREAD_h -D_SYS__PTHREADTYPES_H_'
    AFLAGS  = ' -c' + DEVICE + ' -x assembler-with-cpp -D__ASSEMBLY__'
    LFLAGS  = DEVICE + ' -nostartfiles -Wl,--gc-sections,-Map=rtthread.map,-cref,-u,_start -T link.lds' + ' -lsupc++ -lgcc -static'
    CPATH   = ''
    LPATH   = ''

    if BUILD == 'debug':
        CFLAGS += ' -O0 -ggdb'
        AFLAGS += ' -ggdb'
    else:
        CFLAGS += ' -O2 -Os'

    CXXFLAGS = CFLAGS

DUMP_ACTION = OBJDUMP + ' -D -S $TARGET > rtthread.asm\n'
POST_ACTION = OBJCPY + ' -O binary $TARGET Image \n' + SIZE + ' $TARGET \n'
POST_ACTION += "echo '$TARGET' | awk -F '.' '{print $1}' | xargs -I {} cp -v Image {} \n"
POST_ACTION += OBJDUMP + ' -D -S $TARGET > rtthread.asm\n'

M_CFLAGS = CFLAGS + ' -fPIC '
M_CXXFLAGS = CXXFLAGS + ' -fPIC'
M_LFLAGS = DEVICE + CXXFLAGS + ' -Wl,--gc-sections,-z,max-page-size=0x4' +\
' -fPIC -shared -nostartfiles -lgcc -lc'
M_BIN_PATH = r'/media/psf/workspace/sophgo/rt-thread/bsp/cvitek/output/'
