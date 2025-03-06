#include <rtthread.h>
#include "mmio.h"
#include "stdlib.h"
#include "common.h"

#ifdef SOC_TYPE_BM1690_TP

#define TPSYS_MSI_REG_BASE (C920_TPU_SYS_ADDR + 0xCD00000)
#define TPSYS_MSI_INT_CFG_ADDR    (TPSYS_MSI_REG_BASE + 0x800)

#define MSI_IRQ_NUM 16
#define MSI_IRQ_ID 37

static u_int64_t start_time = 0, end_time = 0;


void msi_interrupt_handler(int vector, void *param)
{
	end_time = time_simple();
	mmio_write_32(TPSYS_MSI_REG_BASE + (vector - 37) * 4, 0);
	rt_kprintf("interrupt %d done!!! time %ld ns\n", vector, end_time - start_time);
}
void msi_irq_init(void)
{
	for(int index = 0; index < MSI_IRQ_NUM; index++) {
		rt_kprintf("requst irq %d... \n", index + MSI_IRQ_ID);
		rt_hw_interrupt_umask(index + MSI_IRQ_ID);
		rt_hw_interrupt_install(index + MSI_IRQ_ID, msi_interrupt_handler, RT_NULL, "c920_irq");
	}
	mmio_write_32(TPSYS_MSI_INT_CFG_ADDR , 0xffffffff);
}
static void msi_irq_test(int argc, char**argv)
{
    if(argc != 2)
    {
        rt_kprintf("msi_irq <num> \n");
        return;
    }
    uint64_t irq_index= strtol(argv[1], NULL, 10);
    rt_kprintf("triger msi_irq[%d]\n", irq_index);

    
    start_time = time_simple();
    mmio_write_32(TPSYS_MSI_REG_BASE + irq_index * 4, 0x1);
}

MSH_CMD_EXPORT(msi_irq_test, msi_irq test: msi_irq <0-15>);
#else
void msi_irq_init(void)
{
    ;
}
#endif