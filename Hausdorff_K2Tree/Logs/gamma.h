#ifndef _GAMMA_H
#define _GAMMA_H
#include "../Util/Factory.h"


class Gamma {
  public:
    uint encode(uint* output, uint pos, uint value);
    uint decode(uint* input, uint pos, uint* value);
};

#endif
