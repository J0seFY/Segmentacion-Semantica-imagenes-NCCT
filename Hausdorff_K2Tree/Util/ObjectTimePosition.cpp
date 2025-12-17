/*
 * ObjectTimePosition.cpp
 *
 *  Created on: 13-09-2013
 *      Author: miguel
 */

#include"ObjectTimePosition.h"

ObjectTimePosition * cObjectTimePosition(uint idObj, uint ts, int x, int y) {
	ObjectTimePosition * otp = NULL;
	otp = (ObjectTimePosition *) calloc(1, sizeof(ObjectTimePosition));
	if (otp == NULL) {
		fprintf(stderr, "\n falla calloc en cObjectTimePosition()");
		exit(1);
	}

	otp->idObj = idObj;
	otp->timeStmp = ts;
	otp->x = x;
	otp->y = y;
	return otp;
}

ObjectTimePosition * cObjectTimePosition(uint idObj, uint ts, Point * p) {
	ObjectTimePosition * otp = NULL;
	otp = (ObjectTimePosition *) calloc(1, sizeof(ObjectTimePosition));
	if (otp == NULL) {
		fprintf(stderr, "\n falla calloc en cObjectTimePosition()");
		exit(1);
	}

	otp->idObj = idObj;
	otp->timeStmp = ts;
	otp->x = p->getX();
	otp->y = p->getY();
	return otp;
}
