/*
 * Log.cpp
 *
 *  Created on: 17-09-2013
 *      Author: miguel
 */

#include "LogRP.h"

LogRP::LogRP() {
	bsMov = NULL;
	changes = NULL;
	lenMov = 0;
	lenLog = 0;
}
LogRP::LogRP(uint len) {
	if (len > 0) {
		lenMov = len;
		lenLog = 0;
		//es fundamental construirlo con () al final para que sea puesto en 0.
		bsMov = new uint[uint_len(len, 1)]();
		changes = cLinkedList();
	} else {
		bsMov = NULL;
		changes = NULL;
		lenMov = 0;
		lenLog = 0;

	}
}

LogRP::~LogRP() {
	if (lenMov > 0) {
		delete[] bsMov;
		bsMov = NULL;
		destroyLinkedList(changes);
		free(changes);
		changes = NULL;
	}
}

void LogRP::Init(uint len) {
	if (len > 0) {
		if (bsMov == NULL) {
			bsMov = new uint[uint_len(len, 1)]();
		}
		if (changes == NULL) {
			changes = cLinkedList();
		}

	}
	lenMov = len;
	lenLog = 0;
}
//----------------------------------------------------------------------------
///almacena la posicion x,y  para el objeto o, en la posición i
///PRE: tiene que ser llamada la función en orden creciente por tiempo, no funciona
///si vienen en otro orden.

void LogRP::setChange(int x, int y, uint i) {
	//hay que checkear que este punto efectivamente sea un nuevo movimiento
	//si es el mismo que el anterior, no hay que insertarlo.
	if (bitget(bsMov,i) == 0) {
		int* pAnt = (changes->ultimo!=NULL)?(int*) changes->ultimo->data->pt:NULL;
		if (!(pAnt!=NULL && pAnt[0] == x && pAnt[1] == y)) {
			//es la primera posición en ese instante
			int * p = (int*) malloc(sizeof(int) * 2);
			p[0] = x;
			p[1] = y;
			bitset(bsMov, i);
			add(changes, cElement(p));
			lenLog++;
		}
	} /*else {
	 //se ignoran las posteriores actualizaciones de un instante.
	 }*/
}

size_t LogRP::size() {
	size_t resp = sizeof(LogRP);
	//tamaño mov
	resp += sizeof(uint) * uint_len(lenMov, 1);
	//tamaño lista
	resp += sizeof(LinkedList)
			+ changes->size
					* (sizeof(Node) + sizeof(Element) + 2 * sizeof(int));
	return resp;
}

void LogRP::printMov() {
	fprintf(stderr, "\n");
	for (int i = 0; i < lenMov; i++) {
		fprintf(stderr, "%d", bitget(bsMov, i));
		if (i % 4 == 0) {
			fprintf(stderr, " ");
		}
	}
	fprintf(stderr, "\n");

}

