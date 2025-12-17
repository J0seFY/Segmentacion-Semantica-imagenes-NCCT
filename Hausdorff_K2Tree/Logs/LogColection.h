/*
 * LogColection.h
 *  Created on: 19-10-2011
 *      Author: miguel
 */

#ifndef LOGCOLECTION_H_
#define LOGCOLECTION_H_

#include <stdio.h>
#include <stdlib.h>
#include "../Util/Point.h"
#include "LogRP.h"

class LogColection {
public:
	LogColection(int totObj, int LenLog);
	virtual ~LogColection();
	//almacenar la posición relativa (dx,dy) para el objeto o
	void setChange(uint o, int dx, int dy, uint i);
    LogRP * getLog(uint o);

	int getTotObj() const {
		return totObj;
	}

	size_t getLenLog() const {
		return lenLog;
	}

	size_t size();

private:
	uint totObj;
	size_t lenLog;
	LogRP ** logArray;

};


#endif /* LOGCOLECTION_H_*/
