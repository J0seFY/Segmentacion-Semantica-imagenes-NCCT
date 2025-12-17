/*
 * Factory.cpp
 *
 *  Created on: 30-12-2013
 *      Author: miguel
 */

#include "Factory.h"

BitSequence * BitSequenceFactory(uint * bitmap, size_t len) {
	BitSequence * resp;
	uint param=20;
//	size_t count1 = 0;
//	uint numInt = uint_len(len, 1);
//	for (uint i = 0; i < numInt; i++) {
//		count1 += popcount(bitmap[i]);
//	}
//	double razon = (count1 * 1.0) / (len * 1.0);
//	if (razon > 20 && razon < 80) {
		resp = new BitSequenceRG(bitmap, len, param);
//	} else if (razon > 5 && razon < 95) {
//		resp = new BitSequenceRRR(bitmap, len, 16u);
//	} else {
//		resp = new BitSequenceSDArray(bitmap, len);
//	}
	return resp;
}

BitSequence * PlainBitSequenceFactory(uint * bitmap, size_t len) {
//	return BitSequenceFactory(bitmap,len);
	uint param=20;
	return new BitSequenceRG(bitmap, len, param);
}
