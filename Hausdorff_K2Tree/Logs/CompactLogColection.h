/*
 * RelativePositions.h
 *
 *  Created on: 19-10-2011
 *      Author: miguel
 */

#ifndef COMPACT_LOG_COLECTION_H_
#define COMPACT_LOG_COLECTION_H_

#include <stdio.h>
#include <stdlib.h>
#include "../Util/Point.h"
#include "CompactLog.h"
#include "LogColection.h"

class CompactLogColection {
public:
	CompactLogColection();
	CompactLogColection(LogColection* lc);
	virtual ~CompactLogColection();

	CompactLog * getCompatLog(uint o);
	size_t size();

	uint * getMaxIzMovTo(uint i);

	unsigned int getTotObj() const {
		return totObj;
	}

private:
	uint totObj;
	CompactLog ** logArray;
	size_t len;
	uint * maxPosDx;
	uint * maxPosDy;
	uint * maxNegDx;
	uint * maxNegDy;

};

#endif /* COMPACT_LOG_COLECTION_H_ */
