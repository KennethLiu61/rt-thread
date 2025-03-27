#include <memmap.h>
#include <arch.h>
#include <framework/module.h>
#include <framework/common.h>

void tbox_putc(int ch)
{
        writel(ch, TBOX_FORMAT_OUT);
}

int tbox_getc(void)
{
        return 1;
}

int tbox_init(void)
{
	register_stdio(tbox_getc, tbox_putc);
	return 0;
}

early_init(tbox_init);