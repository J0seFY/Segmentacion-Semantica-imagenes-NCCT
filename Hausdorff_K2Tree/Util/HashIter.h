/*
 * HashIter.h
 *
 *  Created on: 07-12-2012
 *      Author: miguel
 */

#ifndef HASHITER_H_
#define HASHITER_H_

#include <stdlib.h>
#include "HashMap.h"
#include "Element.h"

typedef struct sHashIterator {
	HashMap * theMap;
	uint countVisited;
	uint actual;
} HashIterator;

//constructor del iterador
HashIterator * cHashIterator(HashMap * rs);

//1 si hay más elementos 0 sino los hay.
int moreElements(HashIterator * iter);

//obtiene el siguiente elemento no leido de la lista
Element * nextElement(HashIterator * iter);

//pone el punero actual al primero del resultSet rs
void restart(HashIterator * iter);

//destruye un iterador
void destroyHashIterator(HashIterator * iter);



#endif /* HASHITER_H_ */
