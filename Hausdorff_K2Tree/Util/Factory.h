/*
 * Factory.h
 *
 *  Created on: 30-12-2013
 *      Author: miguel
 */

#ifndef FACTORY_H_
#define FACTORY_H_

#include <BitSequence.h>
//plain bitmap
#include <BitSequenceRG.h>
//compressed bitmap
#include <BitSequenceRRR.h>
//very sparce bitmap
#include <BitSequenceSDArray.h>
#define ERROR_BITSEQUENCE (-1)

using namespace cds_static;

using namespace std;

BitSequence * BitSequenceFactory(uint * bitmap, size_t len);
BitSequence * PlainBitSequenceFactory(uint * bitmap, size_t len);

#endif /* FACTORY_H_ */
