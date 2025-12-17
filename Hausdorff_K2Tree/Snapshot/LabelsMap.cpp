/*
 * LabelsMap.cpp
 *
 *  Created on: 03-09-2013
 *      Author: miguel
 */

#include "LabelsMap.h"
#include "../Util/Factory.h"

LabelsMap::LabelsMap(LinkedList * list, size_t totObj) {
	//funciona si no hay listas vacías solamente.
	//la lista tiene en count el número total de elementos
	//en las diferentes listas.
	//totObj es el total de objetos del sistema, este parámetro
	//viene dado externamente.  Puede ocurrir que uno o más objetos
	//no formen parte del k2tree, pero la permutación se debe hacer
	//con todos los objetos.  En este caso se crea una lista adicional
	//que contendrá a todos aquellos objetos que no están en el k2tree,
	//solo por completitud.
	uint * bs = new uint[uint_len(totObj, 1)]();

	//para marcar los objetos insertados.
	//este arreglo es solo temporal.
	uint * objSet = new uint[uint_len((uint) totObj, (size_t) 1)]();

	//conversión de la colección de listas list a arreglo de enteros
	ListIterator * iter = cListIterator(list);
	LinkedList * subList;
	ListIterator * subIter;
	uint *arreglo = new uint[totObj]();                //inicializado en 0
	size_t last = 0;
	size_t oid = 0;
	while (moreElements(iter)) {
		subList = (LinkedList*) (get(nextElement(iter)));
		subIter = cListIterator(subList);
		//marco el inicio de la lista
		bitset(bs, last);
		while (moreElements(subIter)) {
			//obtengo el valor de la lista
			oid = *((uint*) (get(nextElement(subIter))));
			bitset(objSet, oid);
			arreglo[last++] = oid;
		}
		destroyListIterator(subIter);
		free(subIter);
		subIter = NULL;
	}
	//completamos la lista de oid poniendo los faltantes como última lista
	bitset(bs, last);
	//creamos un BitSequenceOneLevelRank para recorrer los ceros de objSet
	//este bitmap es temporal asi que no es necesario compactarlo.
	BitSequence * objSetRank = PlainBitSequenceFactory(objSet, totObj);
	oid = 0;
	size_t totCeros = objSetRank->countZeros();
	for (size_t i = 0; i < totCeros; i++) {
		oid = objSetRank->selectNext0(oid);
		arreglo[last++] = oid++;
	}

	//inicializando los atributos del objeto.
	listMark = BitSequenceFactory(bs, totObj);
	delete[] bs;
	bs = NULL;
	//Nota, para t= log log n se consigue n log n + o(n log n)bits de espacio.
	uint t = ceil(log2(log2(list->count * 1.0)));
	//donde dice 1u debe ser t, pero primero hay que corregir PermutationMRRR en libcds
	permutation = new PermMRRR(arreglo, totObj, t);
	//NO es seguro elminar arreglo, este se elimina junto con la permutacion.
	//delete[] arreglo;
	delete objSetRank;
	objSetRank = NULL;
	delete[] objSet;
	objSet = NULL;
	destroyListIterator(iter);
	free(iter);
	iter = NULL;
}
uint * LabelsMap::getObjects(uint listID, uint &len) {
	size_t ini = listMark->select1(listID);
	if (ini == ERROR_BITSEQUENCE) {
		len = 0;
		return NULL;
	}
	size_t fin = listMark->selectNext1(ini + 1);
	fin = (fin == ERROR_BITSEQUENCE) ? listMark->getLength() : fin;
	len = fin - ini;
	uint * resp = NULL;
	if (len > 0) {
		resp = new uint[len];
		int count=0;
		for (size_t i = ini; i < fin; i++) {
			resp[count++] = permutation->Access(i);
		}
	}
	return resp;
}
uint * LabelsMap::getUpToKObjects(uint listID, uint &k) {
	size_t ini = listMark->select1(listID);
	if (ini == ERROR_BITSEQUENCE) {
		k = 0;
		return NULL;
	}
	size_t fin = listMark->selectNext1(ini + 1);
	fin = (fin == ERROR_BITSEQUENCE) ? listMark->getLength() : fin;
	uint len = fin - ini;
	uint * resp = NULL;
	if (len > 0) {
		resp = new uint[len];
		int count = 0;
		for (size_t i = ini; i < fin && count < k; i++, count++) {
			resp[count] = permutation->Access(i);
		}
		k = count;
	}
	return resp;
}
void LabelsMap::getObjects(uint listID, uint x, uint y, LinkedList * result) {
	size_t ini = listMark->select1(listID);
	if (ini == ERROR_BITSEQUENCE)
		return;
	size_t fin = listMark->selectNext1(ini + 1);
	fin = (fin == ERROR_BITSEQUENCE) ? listMark->getLength() : fin;

	uint oid;
	for (size_t i = ini; i < fin; i++) {
		oid = permutation->Access(i);
		ObjectTimePosition * obj = cObjectTimePosition(oid, 0, x, y);
		add(result, cElementKey(obj->idObj, obj));
	}
}

void LabelsMap::getObjects(uint listID, uint x, uint y, uint * &Oid, uint * &X,
							uint * &Y, uint &n) {
	size_t ini = listMark->select1(listID);
	if (ini == ERROR_BITSEQUENCE)
		return;
	size_t fin = listMark->selectNext1(ini + 1);
	fin = (fin == ERROR_BITSEQUENCE) ? listMark->getLength() : fin;

	uint oid;
	for (size_t i = ini; i < fin; i++) {
		oid = permutation->Access(i);
		Oid[n] = oid;
		X[n] = x;
		Y[n] = y;
		n++;
	}
}
size_t LabelsMap::getListID(size_t OID) {
	size_t oidPos = permutation->Reverse(OID);
	size_t listID = listMark->rank1(oidPos);
	return listID;
}

size_t LabelsMap::getSize() {
	return listMark->getSize() + permutation->GetSize() + sizeof(LabelsMap);
}

LabelsMap::~LabelsMap() {
	delete listMark;
	listMark = NULL;
	delete permutation;
	permutation = NULL;
}
