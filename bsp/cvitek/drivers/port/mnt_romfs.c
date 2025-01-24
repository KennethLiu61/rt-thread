/*
 * Copyright (c) 2006-2023, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <rtthread.h>

#ifdef RT_USING_DFS
#include <dfs_fs.h>

#define DBG_TAG "app.filesystem"
#define DBG_LVL DBG_LOG
#include <rtdbg.h>

extern const struct romfs_dirent romfs_root;

int mount_init(void)
{
    /* romfs 挂载在 / 下 */
    if(dfs_mount(RT_NULL, "/", "rom", 0, &romfs_root) != 0)
    {   
        LOG_E("rom mount to '/' failed!");
    }   

    /* tmpfs 挂载在 /tmp 下 */
    if (dfs_mount(RT_NULL, "/tmp", "tmp", 0, NULL) != 0)
    {   
        rt_kprintf("Dir /tmp mount failed!\n");
        return -1; 
    }   	

    return RT_EOK;
}
INIT_ENV_EXPORT(mount_init);

#endif /* RT_USING_DFS */
