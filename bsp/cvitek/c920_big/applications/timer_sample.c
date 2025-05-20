#if 0 // DYNAMIC_CREATE_TIMER

#include <rtthread.h>

/* 定时器的控制块 */
static rt_timer_t timer1;
static rt_timer_t timer2;
static rt_uint8_t cnt = 0;

/* 定时器 1 超时函数 */
static void timeout1(void *parameter)
{
    rt_kprintf("periodic timer is timeout %d\n", cnt);

    /* 运行第 10 次，停止周期定时器 */
    if (cnt++>= 9)
    {
        /* 清除计数值 */
        cnt = 0;
        rt_timer_stop(timer1);
        rt_kprintf("periodic timer was stopped! \n");
    }
}

/* 定时器 2 超时函数 */
static void timeout2(void *parameter)
{
    rt_kprintf("one shot timer is timeout\n");
}

/* 检查，清理上次的定时器 */
static void check_timer_exist(void)
{
    if (timer1 != RT_NULL)
    {
        rt_timer_delete(timer1);
        timer1 = RT_NULL;
    }
    if (timer2 != RT_NULL)
    {
        rt_timer_delete(timer2);
        timer2 = RT_NULL;
    }
}

int timer_sample(void)
{
    /* 检查，清理上次的定时器 */
    check_timer_exist();

    /* 创建定时器 1  周期定时器 */
    timer1 = rt_timer_create("timer1", timeout1,
                             RT_NULL, 100,
                             RT_TIMER_FLAG_PERIODIC);

    /* 启动定时器 1 */
    if (timer1 != RT_NULL) rt_timer_start(timer1);

    /* 创建定时器 2 单次定时器 */
    timer2 = rt_timer_create("timer2", timeout2,
                             RT_NULL,  30,
                             RT_TIMER_FLAG_ONE_SHOT);

    /* 启动定时器 2 */
    if (timer2 != RT_NULL) rt_timer_start(timer2);
    return 0;
}

/* 导出到 msh 命令列表中 */
MSH_CMD_EXPORT(timer_sample, timer sample);

int set_timer_control_sample(void)
{
    rt_timer_control(timer1, RT_TIMER_CTRL_SET_TIME, &(rt_uint32_t){1000});
    if (timer1 != RT_NULL) rt_timer_start(timer1);

    return 0;
}
MSH_CMD_EXPORT(set_timer_control_sample, timer control sample);
#else   //static create timer
#include <rtthread.h>

/* 定时器的控制块 */
static struct rt_timer timer1;
static struct rt_timer timer2;
static int cnt = 0;

/* 定时器 1 超时函数 */
static void timeout1(void* parameter)
{
    rt_kprintf("periodic timer is timeout\n");
    /* 运行 10 次 */
    if (cnt++>= 9)
    {
        cnt = 0;
        rt_timer_stop(&timer1);
    }
}

/* 定时器 2 超时函数 */
static void timeout2(void* parameter)
{
    rt_kprintf("one shot timer is timeout\n");
}

/* 检查，清理上次的定时器 */
static void check_timer_exist(void)
{
   if (rt_object_find("timer1", RT_Object_Class_Timer) != RT_NULL)
   {
       rt_timer_detach(&timer1);
   }
   if (rt_object_find("timer2", RT_Object_Class_Timer) != RT_NULL)
   {
       rt_timer_detach(&timer2);
   }
}

int timer_static_sample(void)
{
    /* 检查，清理上次的定时器 */
    check_timer_exist();
    /* 初始化定时器 */
    rt_timer_init(&timer1, "timer1",  /* 定时器名字是 timer1 */
                    timeout1, /* 超时时回调的处理函数 */
                    RT_NULL, /* 超时函数的入口参数 */
                    10, /* 定时长度，以 OS Tick 为单位，即 10 个 OS Tick */
                    RT_TIMER_FLAG_PERIODIC); /* 周期性定时器 */
    rt_timer_init(&timer2, "timer2",   /* 定时器名字是 timer2 */
                    timeout2, /* 超时时回调的处理函数 */
                      RT_NULL, /* 超时函数的入口参数 */
                      30, /* 定时长度为 30 个 OS Tick */
                    RT_TIMER_FLAG_ONE_SHOT); /* 单次定时器 */

    /* 启动定时器 */
    rt_timer_start(&timer1);
    rt_timer_start(&timer2);
    return 0;
}
/* 导出到 msh 命令列表中 */
MSH_CMD_EXPORT(timer_static_sample, timer_static sample);

int set_timer_control_sample(void)
{
    rt_timer_control(&timer1, RT_TIMER_CTRL_SET_TIME, &(rt_uint32_t){1000});
    if (&timer1 != RT_NULL) rt_timer_start(&timer1);

    return 0;
}
MSH_CMD_EXPORT(set_timer_control_sample, timer control sample);

#endif