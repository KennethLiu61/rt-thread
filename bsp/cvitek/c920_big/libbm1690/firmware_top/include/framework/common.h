#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <timer.h>

/* stdinput:
 *	return none-nagtive number means input character
 *	return nagtive number means no input character in buffer
 * stdout:
 *	output character to stdout
 */
void register_stdio(int (*stdinput)(void), void (*stdoutput)(int));

int stdio_input(void);
void stdio_output(int ch);
int stdout_ready(void);

/* log system */
#ifdef DEBUG
#define pr_debug(fmt, ...)	printf(fmt, ##__VA_ARGS__)
#else
#define pr_debug(fmt, ...)	do {} while (0)
#endif
#define pr_info(fmt, ...)                                  \
    do                                                     \
    {                                                      \
        while (readl(0x260501d8ULL))                       \
                timer_udelay(1);                           \
        printf("[%d]" fmt, CONFIG_CORE_ID, ##__VA_ARGS__); \
        writel(0, 0x260501d8ULL);                          \
    } while (0)
#define pr_warn(fmt, ...)	printf(fmt, ##__VA_ARGS__)
#define pr_err(fmt, ...)	printf(fmt, ##__VA_ARGS__)

#define ARRAY_SIZE(a)	(sizeof(a) / sizeof((a)[0]))
#define ROUND_UP(x, align)	(((x) + ((align) - 1)) & ~((align) - 1))

#endif
