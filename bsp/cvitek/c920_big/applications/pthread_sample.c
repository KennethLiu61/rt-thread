#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

extern void *rvv_memcpy(void* dest, const void* src, size_t n);

/* 线程控制块 */
static pthread_t tid1;
static pthread_t tid2;
static char tmp_buf[10] = {"hello\n"};

/* 函数返回值检查 */
static void check_result(char* str,int result)
{
    if (0 == result)
    {
        printf("%s successfully!\n",str);
    }
    else
    {
        printf("%s failed! error code is %d\n",str,result);
    }
}

/* 线程入口函数 */
static void* thread_entry(void* parameter)
{
    int count = 0;
    int no = (uint64_t) parameter; /* 获得线程的入口参数 */
    char buf[30] = {0};
    char src_buf[30] = {0};
    rvv_memcpy(buf, tmp_buf, 10);
    rt_kprintf("[%d]buf = %s\n", no, buf);

    while (count <= 5)
    {
        /* 打印输出线程计数值 */
        sprintf(src_buf, "thread%d count: %d\n", no, count ++);
        rvv_memcpy(buf, src_buf, sizeof(src_buf));
        printf("%s", buf);

        sleep(2);    /* 休眠 2 秒 */
        // rt_thread_mdelay(2000);
    }
}

/* 用户应用入口 */
int rvv_test(void)
{
    int result;

    pthread_attr_t attr;
    size_t stack_size = 0;
    pthread_attr_init(&attr);
    pthread_attr_getstacksize(&attr, &stack_size);
    printf("the stacksize = 0x%x\n", stack_size);
    pthread_attr_setstacksize(&attr, 4096);

    /* 创建线程 1, 属性为默认值，入口函数是 thread_entry，入口函数参数是 1 */
    result = pthread_create(&tid1, &attr,thread_entry,(void*)1);
    check_result("thread1 created", result);
#if 0
    rt_thread_t tid = RT_NULL;
    // tid = rt_thread_create("thread1",
    // thread_entry, (void *)1,
    // RT_MAIN_THREAD_STACK_SIZE,
    // RT_MAIN_THREAD_PRIORITY, 20);
    // if(tid != NULL)
    //     rt_thread_startup(tid);
#endif

    /* 创建线程 2, 属性为默认值，入口函数是 thread_entry，入口函数参数是 2 */
    result = pthread_create(&tid2, &attr,thread_entry,(void*)2);
    check_result("thread2 created", result);

    return 0;
}

MSH_CMD_EXPORT(rvv_test, rvv sample test)
