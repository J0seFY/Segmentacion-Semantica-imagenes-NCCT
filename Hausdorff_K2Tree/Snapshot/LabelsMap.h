/*
 * LabelsMap.h
 *
 *  Created on: 03-09-2013
 *      Author: miguel
 */

#ifndef LABELSMAP_H_
#define LABELSMAP_H_




#include "../Util/LinkedList.h"
#include "../Util/ListIterator.h"
#include "../Util/HashMap.h"
#include "../Util/ObjectTimePosition.h"
#include "../Util/Factory.h"
#include "../Snapshot/PermMRRR.h"
#include <math.h>

#include <BitSequence.h>
using namespace cds_static;



//using namespace std;



class LabelsMap {
public:
	LabelsMap(LinkedList * labelsLists, size_t totObj);
	void   getObjects(uint listID,uint x, uint y, LinkedList* result);
	void   getObjects(uint listID,uint x, uint y, uint * &Oid, uint * &X, uint * &Y, uint &n);
	uint * getObjects(uint listID, uint &len);
	uint * getUpToKObjects(uint listID, uint &k);
	size_t getListID(size_t OID);
	size_t getSize();

	virtual ~LabelsMap();

private:
	BitSequence *listMark;
	PermMRRR *permutation;

};

#endif /* LABELSMAP_H_ */
