/*
 * Log.h
 *
 *  Created on: 17-09-2013
 *      Author: miguel
 */

#ifndef COMPACT_LOG_H_
#define COMPACT_LOG_H_
#include "Codec.h"
#include "../Logs/LogRP.h"
#include "../Util/Point.h"
#include "../Util/LinkedList.h"
#include "../Util/ObjectTimePosition.h"
#include "../Snapshot/AreaExpandida.h"
#include <stdlib.h>


#include <BitSequence.h>
using namespace cds_static;

using namespace std;

class CompactLog {
public:
	CompactLog(LogRP * logRP);

	virtual ~CompactLog();
	//entrega un verdadero o un falso, si el punto están en el área, durante
	//el intervalo de tiempo dado.
	int timeIntervalTest(uint x, uint y, uint t1, uint t2, AreaExpandida area );
	int timeSliceTest(uint x, uint y, uint i, AreaExpandida area);
	Point * getSumMov(uint j);
	//arreglo de puntos
	uint** getSumMov();

	size_t size();
	Point * getTrajectory(uint oid, Point * pBase, uint tBase, uint i,
								uint j, LinkedList * resp);
	//indica el número de muestras que contiene la trayectoria entre i y j.
	uint countSamples(uint i, uint j);
	uint getLenTimeInterval();
private:
	BitSequence * mov;
	uint * cdx;
	uint * cdy;
	uint lenX;
	uint lenY;
friend class CompactLogIterator;
};

#endif /* COMPACT_LOG_H_ */
