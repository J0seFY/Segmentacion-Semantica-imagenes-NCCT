/*
 * ObjectTimePosition.h
 *
 *  Created on: 13-09-2013
 *      Author: miguel
 */

#ifndef OBJECTTIMEPOSITION_H_
#define OBJECTTIMEPOSITION_H_
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "Point.h"

typedef struct sObjectTimePosition {
	uint idObj;
	uint timeStmp;
	long int x;
	long int y;
} ObjectTimePosition;


ObjectTimePosition * cObjectTimePosition(uint idObj, uint ts, int x, int y);

ObjectTimePosition * cObjectTimePosition(uint idObj, uint ts, Point * p);

#endif /* OBJECTTIMEPOSITION_H_ */
