#include <rtthread.h>
#define THREAD_PRIORITY 15
#define THREAD_STACK_SIZE 1024
#define THREAD_TIMESLICE 5
static rt_thread_t tid1 = RT_NULL;

/* 线 程 1 的 入 口 函 数 */
static void thread1_entry(void *parameter)
{
    rt_uint32_t count = 0;
    while (1)
    {
    /* 线 程 1 采 用 低 优 先 级 运 行， 一 直 打 印 计 数 值 */
    rt_kprintf("thread1 count: %d\n", count ++);
    rt_thread_mdelay(500);
	if(count >= 50)
		break;
    }
}
rt_align(RT_ALIGN_SIZE)
static char thread2_stack[2048];
static struct rt_thread thread2;
/* 线 程 2 入 口 */
static void thread2_entry(void *param)
{
    rt_uint32_t count = 0;
    /* 线 程 2 拥 有 较 高 的 优 先 级， 以 抢 占 线 程 1 而 获 得 执 行 */
    for (count = 0; count < 10 ; count++)
    {
    /* 线 程 2 打 印 计 数 值 */
    rt_kprintf("thread2 count: %d\n", count);
    }
    rt_kprintf("thread2 exit\n");
    /* 线 程 2 运 行 结 束 后 也 将 自 动 被 系 统 脱 离 */
}
/* 线 程 示 例 */
int thread_sample(void)
{
    /* 创 建 线 程 1， 名 称 是 thread1， 入 口 是 thread1_entry*/
    tid1 = rt_thread_create("thread1",
    thread1_entry, RT_NULL,
    RT_MAIN_THREAD_STACK_SIZE,
    RT_MAIN_THREAD_PRIORITY, 20);
    /* 如 果 获 得 线 程 控 制 块， 启 动 这 个 线 程 */
    if (tid1 != RT_NULL)
    rt_thread_startup(tid1);

    /* 初 始 化 线 程 2， 名 称 是 thread2， 入 口 是 thread2_entry */
    rt_thread_init(&thread2,
    "thread2",
    thread2_entry,
    RT_NULL,
    &thread2_stack[0],
    sizeof(thread2_stack),
    THREAD_PRIORITY - 1, THREAD_TIMESLICE);
    rt_thread_startup(&thread2);

    return 0;
}
/* 导 出 到 msh 命 令 列 表 中 */
MSH_CMD_EXPORT(thread_sample, thread sample);
