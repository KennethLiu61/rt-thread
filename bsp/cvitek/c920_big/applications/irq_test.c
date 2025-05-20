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

//测试禁止调度或者禁止中断，是否真的有效屏蔽中断。
//结论：禁止中断有效屏蔽中断，但是禁止调度不会屏蔽中断。
void my_spinlock_test(void)
{
    rt_kprintf("my_spinlock_test \n");
    struct rt_spinlock lock;
    rt_base_t level;
    rt_spin_lock_init(&lock);

    rt_enter_critical();        //只禁止调度，不会禁止中断
    // level = rt_spin_lock_irqsave(&lock); //禁止中断也禁止调度
    rt_kprintf("spin lock irq save \n");

    //trigger irq
    mmio_write_32(TPSYS_MSI_REG_BASE, 0x1);
    for(int i = 0; i < 100000000; i++)
        ;

    rt_kprintf("spin unlock irq restore \n");
    // rt_spin_unlock_irqrestore(&lock, level);
    rt_exit_critical_safe(level);
}
MSH_CMD_EXPORT(my_spinlock_test, test spin_lock_irqsave func);
#else
void msi_irq_init(void)
{
    ;
}
#endif