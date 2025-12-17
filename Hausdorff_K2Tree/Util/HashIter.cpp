/*
 * HashIter.cpp
 *
 *  Created on: 07-12-2012
 *      Author: miguel
 */
#include "HashIter.h"

//constructor del iterador
HashIterator * cHashIterator(HashMap * rs) {
	if (rs == NULL)
		return NULL;

	HashIterator * iter = (HashIterator *) malloc(sizeof(HashIterator));
	iter->theMap = rs;
	iter->countVisited = 0;
	iter->actual = -1;
	return iter;
}

//1 si hay más elementos 0 sino los hay.
int moreElements(HashIterator * iter) {
	//PRE: iter no es nulo
	if ((iter == NULL) || (iter->theMap == NULL)) {
		return 0;
	}
	return (iter->countVisited < iter->theMap->size);
}

//obtiene el siguiente elemento no leido del hashmap
Element * nextElement(HashIterator * iter) {
	while(moreElements(iter)&&(iter->theMap->map[++(iter->actual)]==NULL));
    if(moreElements(iter)){
    	iter->countVisited++;
    	return iter->theMap->map[iter->actual];
    }
	return NULL;
}

//pone el punero actual al primero del resultSet rs
void restart(HashIterator * iter) {
	iter->actual = -1;
	iter->countVisited = 0;
}

//destruye un iterador
void destroyHashIterator(HashIterator * iter) {
	iter->theMap = NULL;
	free(iter);
	iter=NULL;
}

