// SPDX-License-Identifier: GPL-2.0

#include <stdio.h>
#include <stdlib.h>

#include "md5.h"
// #define USING_TP_DEBUG

void show_md5(unsigned char md5[])
{
#ifdef USING_TP_DEBUG
	printf("md5 is ");
	for (int i = 0; i < MD5SUM_LEN; i++)
		printf("%02x", md5[i]);
	printf("\n");
#endif
}
