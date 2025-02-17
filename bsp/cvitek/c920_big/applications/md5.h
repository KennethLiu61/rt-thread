/* SPDX-License-Identifier: GPL-2.0 */

#ifndef MD5_H
#define MD5_H

#define MD5SUM_LEN 16

struct MD5_CTX {
	unsigned int count[2];
	unsigned int state[4];
	unsigned char buffer[64];
};

#define F(x, y, z)  ((x & y) | (~x & z))
#define G(x, y, z)  ((x & z) | (y & ~z))
#define H(x, y, z)  (x ^ y ^ z)
#define I(x, y, z)  (y ^ (x | ~z))
#define ROTATE_LEFT(x, n)   ((x << n) | (x >> (32 - n)))
#define FF(a, b, c, d, x, s, ac) \
		{ \
		a += F(b, c, d) + x + ac; \
		a = ROTATE_LEFT(a, s); \
		a += b; \
		}
#define GG(a, b, c, d, x, s, ac) \
		{ \
		a += G(b, c, d) + x + ac; \
		a = ROTATE_LEFT(a, s); \
		a += b; \
		}
#define HH(a, b, c, d, x, s, ac) \
		{ \
		a += H(b, c, d) + x + ac; \
		a = ROTATE_LEFT(a, s); \
		a += b; \
		}
#define II(a, b, c, d, x, s, ac) \
		{ \
		a += I(b, c, d) + x + ac; \
		a = ROTATE_LEFT(a, s); \
		a += b; \
		}

void MD5Init(struct MD5_CTX *context);
void MD5Update(struct MD5_CTX *context, unsigned char *input, unsigned int inputlen);
void MD5Final(struct MD5_CTX *context, unsigned char digest[16]);
void MD5Transform(unsigned int state[4], unsigned char *block);
void MD5Encode(unsigned char *output, unsigned int *input, unsigned int len);
void MD5Decode(unsigned int *output, unsigned char *input, unsigned int len);
void read_md5(unsigned char *file_path, unsigned char *md5sum);
void calc_md5(unsigned char *data, unsigned long len, unsigned char *md5sum);
void show_md5(unsigned char md5[]);
#endif
