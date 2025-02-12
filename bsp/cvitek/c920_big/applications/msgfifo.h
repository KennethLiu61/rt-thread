/* SPDX-License-Identifier: GPL-2.0 */

#ifndef __MSGFIFO_H__
#define __MSGFIFO_H__

#include "api.h"

void msgfifo_init(void);
void msgfifo_process(void);
int msgfifo_empty(void);
int msgfifo_read_task(void);
void msgfifo_finish_api(struct task_response *task_response);

#endif