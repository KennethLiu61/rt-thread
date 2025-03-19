#include <rtthread.h>
#include "mmio.h"
#include "stdlib.h"

static void rt_mr(int argc, char**argv)
{
    if(argc != 2)
    {
        rt_kprintf("rt_mr <addr>\n");
        return;
    }
    uint64_t addr = strtol(argv[1], NULL, 16);
    uint32_t reg = mmio_read_32(addr);
    rt_kprintf("read <0x%lx>: 0x%x\n", addr, reg);
}

MSH_CMD_EXPORT(rt_mr, mmio read reg: rt_mr <addr>);

static void rt_mr64(int argc, char**argv)
{
    if(argc != 2)
    {
        rt_kprintf("rt_mr64 <addr>\n");
        return;
    }
    uint64_t addr = strtol(argv[1], NULL, 16);
    uint64_t reg = mmio_read_64(addr);
    rt_kprintf("read <0x%lx>: 0x%lx\n", addr, reg);
}

MSH_CMD_EXPORT(rt_mr64, mmio read reg: rt_mr64 <addr>);

static void rt_mw64(int argc, char**argv)
{
    if(argc != 3)
    {
        rt_kprintf("rt_mw64 <addr> <val>\n");
        return;
    }
    uint64_t addr = strtol(argv[1], NULL, 16);
    uint64_t reg = strtol(argv[2], NULL, 16);
    mmio_write_64(addr, reg);
    rt_kprintf("write <0x%lx>: 0x%lx\n", addr, reg);
}

MSH_CMD_EXPORT(rt_mw64, mmio write reg: rt_mw64 <addr>);

static void rt_mw(int argc, char**argv)
{
    if(argc != 3)
    {
        rt_kprintf("rt_mw <addr> <val>\n");
        return;
    }
    uint64_t addr = strtol(argv[1], NULL, 16);
    uint32_t reg = strtol(argv[2], NULL, 16);
    mmio_write_32(addr, reg);
    rt_kprintf("write <0x%lx>: 0x%x\n", addr, reg);
}

MSH_CMD_EXPORT(rt_mw, mmio write reg: rt_mw <addr>);

static void rt_dump_reg(int argc, char**argv)
{
    if(argc != 3) {
        rt_kprintf("rt_dump_reg <addr> <num>\n");
    }
    uint64_t addr = strtol(argv[1], NULL, 16);
    uint32_t num = strtol(argv[2], NULL, 10);
    for(int i = 0; i < num; i++) {
        uint32_t reg = mmio_read_32(addr + i * 4);
        rt_kprintf("0x%x: 0x%x\n", addr + i * 4, reg);
    }
}
MSH_CMD_EXPORT(rt_dump_reg, mmio dump reg: rt_dump_reg <addr> <num>);