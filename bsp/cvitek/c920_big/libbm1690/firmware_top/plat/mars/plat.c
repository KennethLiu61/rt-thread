#include <stdint.h>
#include <framework/module.h>
#include <framework/common.h>

static int mars_init(void)
{
	pr_debug("platform init\n");
	return 0;
}

plat_init(mars_init);
