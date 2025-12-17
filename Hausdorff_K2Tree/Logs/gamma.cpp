#include "gamma.h"

uint Gamma::encode(uint* output, uint pos, uint value) {
	value++;
	register uint code, gammaCodePos, p;
        code = value;
        p = 0;
        while(code) {
          code >>= 1;
          p++;
        }

        gammaCodePos = pos+p-1;

        code = value;
        if( ((gammaCodePos%32) + p) > 32 ) {
          output[gammaCodePos/32] |= (code>>((gammaCodePos%32)+p-32));
          output[gammaCodePos/32+1] = (code<<(64-(gammaCodePos%32)-p));
        } else {
          output[gammaCodePos/32] |= (code<<(32-p-(gammaCodePos%32)));
        }

        return 2*p-1;
}

uint Gamma::decode(uint* input, uint pos, uint* value) {
  register uint code, p;
  register uint gammaCodePos = pos;

  code = (input[gammaCodePos/32] << (gammaCodePos%32)) |
                  ((gammaCodePos%32 != 0) * (input[gammaCodePos/32+1] >> (32-(gammaCodePos%32))));

  p = 1;
  while(!(code & 0x80000000)) {
          code <<= 1;
          p++;
  }

        gammaCodePos += p-1;
  code = (input[gammaCodePos/32] << (gammaCodePos%32)) |
                  ((gammaCodePos%32 != 0) * (input[gammaCodePos/32+1] >> (32-(gammaCodePos%32))));
        code >>= 32-p;

  *value = code-1;
  return 2*p-1;
}

