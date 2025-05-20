#if 0
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

/* 共享资源结构体 */
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int ready;          // 条件变量关联的状态
    int trigger_count;  // 触发计数器
} shared_data_t;

/* 线程参数 */
typedef struct {
    int id;
    shared_data_t *data;
} thread_args_t;

/* 超时等待线程 */
void* timed_wait_thread(void *arg) {
    thread_args_t *args = (thread_args_t*)arg;
    shared_data_t *data = args->data;
    
    rt_kprintf("timed_wait_thread start\n");
    pthread_mutex_lock(&data->mutex);
    
    // 计算超时时间：当前时间 + 2秒
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 2;
    
    printf("[TimedWait-%d] 等待条件 (超时时间: %ld秒)\n", args->id, ts.tv_sec);
    
    // 循环等待条件（防止虚假唤醒）
    while (data->ready == 0) {
        // int rc = pthread_cond_timedwait(&data->cond, &data->mutex, &ts);
        int rc = 0;
        if (rc == ETIMEDOUT) {
            printf("[TimedWait-%d] 超时!\n", args->id);
            break;
        } else if (rc != 0) {
            perror("pthread_cond_timedwait");
            exit(1);
        }
    }
    
    if (data->ready) {
        printf("[TimedWait-%d] 条件满足!\n", args->id);
    }
    
    pthread_mutex_unlock(&data->mutex);
    rt_kprintf("pthread_cond_timedwait end\n");
    return NULL;
}

/* 普通条件等待线程 */
void* cond_wait_thread(void *arg) {
    thread_args_t *args = (thread_args_t*)arg;
    shared_data_t *data = args->data;
    
    rt_kprintf("cond_wait_thread start\n");
    pthread_mutex_lock(&data->mutex);
    printf("[CondWait-%d] 进入等待\n", args->id);
    
    // 循环等待条件
    while (data->ready == 0) {
        // pthread_cond_wait(&data->cond, &data->mutex);
    }
    
    printf("[CondWait-%d] 被唤醒!\n", args->id);
    data->trigger_count++;
    
    pthread_mutex_unlock(&data->mutex);
    rt_kprintf("pthread_cond_wait end\n");
    return NULL;
}

/* 触发条件线程 */
void* trigger_thread(void *arg) {
    shared_data_t *data = (shared_data_t*)arg;
    
    rt_kprintf("trigger_thread start\n");
    sleep(1); // 给等待线程足够时间进入等待
    
    pthread_mutex_lock(&data->mutex);
    data->ready = 1;
    printf("[Trigger] 发送条件信号...\n");
    
    // 测试两种触发方式（二选一）：
    // pthread_cond_signal(&data->cond);  // 唤醒单个线程
    pthread_cond_broadcast(&data->cond);   // 唤醒所有线程
    
    pthread_mutex_unlock(&data->mutex);
    rt_kprintf("pthread_cond_signal end\n");
    return NULL;
}

int pthread_cond_test(void)
{
    shared_data_t data = {
        .mutex = PTHREAD_MUTEX_INITIALIZER,
        .cond = PTHREAD_COND_INITIALIZER,
        .ready = 0,
        .trigger_count = 0
    };

    pthread_t threads[5];
    thread_args_t args[4];
    pthread_attr_t attr;
    struct sched_param param;

    param.sched_priority = 20; 

    // 设置线程属性为分离状态
    pthread_attr_init(&attr);
    pthread_attr_setschedpolicy(&attr, SCHED_RR);
    pthread_attr_setschedparam(&attr, &param);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&attr, 1024 * 4);
    
    // 启动两个普通条件等待线程
    // printf("启动两个普通条件等待线程\n");
    // for (int i = 0; i < 2; i++) {
    //     args[i].id = i+1;
    //     args[i].data = &data;
    //     pthread_create(&threads[i], &attr, cond_wait_thread, &args[i]);
    // }
    
    // 启动两个超时等待线程
    printf("启动两个超时等待线程\n");
    for (int i = 2; i < 4; i++) {
        args[i].id = i-1;
        args[i].data = &data;
        pthread_create(&threads[i], &attr, timed_wait_thread, &args[i]);
    }
    
    // 启动触发线程
    printf("启动触发线程\n");
    pthread_create(&threads[4], &attr, trigger_thread, &data);
    
    // 等待所有线程结束
    // printf("等待所有线程结束\n");
    // for (int i = 0; i < 5; i++) {
    //     pthread_join(threads[i], NULL);
    // }
    
    // // 验证触发次数
    // printf("\n测试结果:\n");
    // printf("触发次数: %d (预期: 2次普通唤醒)\n", data.trigger_count);
    
    // // 清理资源
    // pthread_mutex_destroy(&data.mutex);
    // pthread_cond_destroy(&data.cond);
    
    return 0;
}
/* 导出到 msh 命令列表中 */
MSH_CMD_EXPORT(pthread_cond_test, pthread_cond_wait sample for test);
#else
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

/* 静态方式初始化一个互斥锁和一个条件变量 */
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

/* 指向线程控制块的指针 */
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

/* 生产者生产的结构体数据，存放在链表里 */
struct node
{
    int n_number;
    struct node* n_next;
};
struct node* head = NULL; /* 链表头, 是共享资源 */

/* 消费者线程入口函数 */
static void* consumer(void* parameter)
{
    struct node* p_node = NULL;

    pthread_mutex_lock(&mutex);    /* 对互斥锁上锁 */

    while (1)
    {
        // 设置超时时间（这里设置为 3 秒后超时）
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);  // 获取当前时间
        ts.tv_sec += 3;                      // 设置超时时间点

        while (head == NULL)    /* 判断链表里是否有元素 */
        {
            #define USE_TIMED_WAIT
            #ifndef USE_TIMED_WAIT
            pthread_cond_wait(&cond,&mutex); /* 尝试获取条件变量 */
            #else
            int rc = pthread_cond_timedwait(&cond, &mutex, &ts);
            if (rc == ETIMEDOUT) 
            {
                // 超时处理逻辑
                printf("[Consumer] timeout...\n");
                
                /* 可选操作：
                   1. 可以继续等待（重新计算 ts）
                   2. 退出线程
                   3. 执行其他恢复操作
                */
                break;  // 这里选择退出等待循环
                // continue;
            } else if (rc != 0) 
            {
                // 其他错误处理
                printf("pthread_cond_timedwait rc = %d\n", rc);
                pthread_mutex_unlock(&mutex);
                return NULL;
            }
            #endif
        }
        /*
        pthread_cond_wait() 会先对 mutex 解锁，
        然后阻塞在等待队列，直到获取条件变量被唤醒，
        被唤醒后，该线程会再次对 mutex 上锁，成功进入临界区。
        */

        p_node = head;    /* 拿到资源 */
        head = head->n_next;    /* 头指针指向下一个资源 */
        /* 打印输出 */
        printf("consume %d\n",p_node->n_number);

        free(p_node);    /* 拿到资源后释放节点占用的内存 */
    }
    pthread_mutex_unlock(&mutex);    /* 释放互斥锁 */
    return 0;
}
/* 生产者线程入口函数 */
static void* product(void* patameter)
{
    int count = 0;
    struct node *p_node;

    while(1)
    {
        /* 动态分配一块结构体内存 */
        p_node = (struct node*)malloc(sizeof(struct node));
        if (p_node != NULL)
        {
            p_node->n_number = count++;
            pthread_mutex_lock(&mutex);    /* 需要操作 head 这个临界资源，先加锁 */

            p_node->n_next = head;
            head = p_node;    /* 往链表头插入数据 */

            pthread_mutex_unlock(&mutex);    /* 解锁 */
            printf("produce %d\n",p_node->n_number);

            pthread_cond_signal(&cond);    /* 发信号唤醒一个线程 */

            sleep(2);    /* 休眠 2 秒 */
        }
        else
        {
            printf("product malloc node failed!\n");
            break;
        }
    }
}

int pthread_cond_test(void)
{
    int result;
    pthread_attr_t attr;
    struct sched_param param;

    param.sched_priority = 20; 

    // 设置线程属性为分离状态
    pthread_attr_init(&attr);
    pthread_attr_setschedpolicy(&attr, SCHED_RR);
    pthread_attr_setschedparam(&attr, &param);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&attr, 1024 * 8);

    /* 创建生产者线程, 属性为默认值，入口函数是 product，入口函数参数为 NULL*/
    result = pthread_create(&tid1,&attr,product,NULL);
    check_result("product thread created",result);

    /* 创建消费者线程, 属性为默认值，入口函数是 consumer，入口函数参数是 NULL */
    result = pthread_create(&tid2,&attr,consumer,NULL);
    check_result("consumer thread created",result);

    return 0;
}

/* 导出到 msh 命令列表中 */
MSH_CMD_EXPORT(pthread_cond_test, pthread_cond_wait sample for test);
#endif

int mytest_addr = 0x0;
int pthread_cond_flag_get(void)
{
    rt_kprintf("mytest_addr = %d\n", mytest_addr);
    return 0;
}
MSH_CMD_EXPORT(pthread_cond_flag_get, pthread_cond_wait sample for test);