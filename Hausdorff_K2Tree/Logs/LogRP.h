/*
 * Log.h
 *
 *  Created on: 17-09-2013
 *      Author: miguel
 */

#ifndef LOG_RP_H_
#define LOG_RP_H_
#include "../Util/Point.h"
#include "../Util/LinkedList.h"
#include "../Util/ListIterator.h"

#include <BitSequenceRG.h>
using namespace cds_static;

using namespace std;

class LogRP {
public:
	LogRP();
	LogRP(uint len);
	virtual ~LogRP();
	void Init(uint len);
	void printMov();

	//--------------------------------------------------------------------------
	// almacenar la posición x,y para el objeto o, en la posición i
	//
	//
	void setChange(int x, int y, uint i);
	ListIterator * getChangeIter(){
		return cListIterator(changes);
	}

	uint* getBsMov() {
		return bsMov;
	}

	size_t getLenLog() const {
		return lenLog;
	}

	size_t getLenMov() const {
		return lenMov;
	}
	size_t size();
private:
	size_t lenMov;
	size_t lenLog;
	uint * bsMov;
	LinkedList * changes;

};

#endif /* LOG_RP_H_ */
