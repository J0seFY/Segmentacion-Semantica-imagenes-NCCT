/*
 * Codec.h
 *
 *  Created on: 17-10-2013
 *      Author: miguel
 */

#ifndef CODEC_RP_H_
#define CODEC_RP_H_
#include "../Util/Factory.h"
//zig-zag coding is a maping 0, -1, 1, -2, 2... to 0,1,2,3,4...
class ZigZagCoder {
public:
	ZigZagCoder();
	virtual ~ZigZagCoder();
	int cod(int value);
	int decod(int value);
};

#endif /* CODEC_H_ */
