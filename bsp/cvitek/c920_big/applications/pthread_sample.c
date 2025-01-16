#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

/* 线程控制块 */
static pthread_t tid1;
static pthread_t tid2;

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
    int no = (int) parameter; /* 获得线程的入口参数 */

    while (1)
    {
        /* 打印输出线程计数值 */
        printf("thread%d count: %d\n", no, count ++);

        sleep(2);    /* 休眠 2 秒 */
    }
}

/* 用户应用入口 */
int pthread_sample_test(void)
{
    int result;

    /* 创建线程 1, 属性为默认值，入口函数是 thread_entry，入口函数参数是 1 */
    result = pthread_create(&tid1,NULL,thread_entry,(void*)1);
    check_result("thread1 created", result);

    /* 创建线程 2, 属性为默认值，入口函数是 thread_entry，入口函数参数是 2 */
    result = pthread_create(&tid2,NULL,thread_entry,(void*)2);
    check_result("thread2 created", result);

    return 0;
}