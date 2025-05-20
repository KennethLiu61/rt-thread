#include <rtthread.h>

#define THREAD_PRIORITY         25
#define THREAD_STACK_SIZE       8192
#define THREAD_TIMESLICE        10

static rt_thread_t tid1 = RT_NULL;

/* 线程 1 的入口函数 */
static void thread1_entry(void *parameter)
{
    rt_uint32_t count = 0;


    for (count = 0; count < 30 ; count++)
    {
        /* 线程 1 采用低优先级运行 */
        rt_kprintf("thread1 count: %d\n", count);
        //不释放调度
        // udelay(1000);
        rt_hw_us_delay(1000);
    }
    rt_kprintf("thread1 exit\n");
    /* 线程 1 运行结束后也将自动被系统脱离 */
}

#if defined(RT_VERSION_CHECK) && (RTTHREAD_VERSION >= RT_VERSION_CHECK(5, 0, 1))
    rt_align(RT_ALIGN_SIZE)
#else
    ALIGN(RT_ALIGN_SIZE)
#endif
static char thread2_stack[8192];
static struct rt_thread thread2;
/* 线程 2 入口 */
static void thread2_entry(void *param)
{
    rt_uint32_t count = 0;

    /* 线程 2 拥有较高的优先级，以抢占线程 1 而获得执行 */
    for (count = 0; count < 30 ; count++)
    {
        /* 线程 2 打印计数值 */
        rt_kprintf("thread2 count: %d\n", count);
        //不释放调度
        // udelay(1000);
        rt_hw_us_delay(1000);
    }
    rt_kprintf("thread2 exit\n");
    /* 线程 2 运行结束后也将自动被系统脱离 */
}

/* 线程示例 */
int thread_prority_sample(void)
{
    /* 创建线程 1，名称是 thread1，入口是 thread1_entry*/
    tid1 = rt_thread_create("thread1",
                            thread1_entry, RT_NULL,
                            THREAD_STACK_SIZE,
                            THREAD_PRIORITY, THREAD_TIMESLICE);

    /* 如果获得线程控制块，启动这个线程 */
    if (tid1 != RT_NULL)
        rt_thread_startup(tid1);

    rt_thread_mdelay(10);
    /* 初始化线程 2，名称是 thread2，入口是 thread2_entry */
    rt_thread_init(&thread2,
                   "thread2",
                   thread2_entry,
                   RT_NULL,
                   &thread2_stack[0],
                   sizeof(thread2_stack),
                   THREAD_PRIORITY, THREAD_TIMESLICE);          //case 2
                //    THREAD_PRIORITY - 1, THREAD_TIMESLICE);   //case 1
    rt_thread_startup(&thread2);

    //测试结果：
    /*
    case 1 : thread2的优先级高于thread1
        thread1虽然先运行，但是运行到thread2开始的时候，就全部执行thread2的内容，
        而且thread2中没有释放调度，所以测试结果就是thread1运行一小会，然后切到thread2运行到它结束后，
        重新回到thread1运行；
    case 2 : 优先级一样的情况下，timeslice一样
        thread1先运行，然后是thread2运行。因为优先级一样，所以会轮流运行。又因为timeslice一样，所以他们执行的时间是一样的；
        例如timeslice=5的情况下，轮流执行2次后，切换到另一个线程；
            timeslice=10的时候，每个线程执行4次后，切换;
    case 3 : 优先级一样的情况下，timeslice不一样，是什么效果
    */
    return 0;
}

/* 导出到 msh 命令列表中 */
MSH_CMD_EXPORT(thread_prority_sample, thread prority sample);
