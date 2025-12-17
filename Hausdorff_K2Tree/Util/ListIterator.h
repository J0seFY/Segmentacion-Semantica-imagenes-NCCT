/*
 * IteradorResultSet.h
 *
 *  Created on: 09-10-2011
 *      Author: miguel
 */

#ifndef LISTITERATOR_H_
#define LISTITERATOR_H_

#include <stdlib.h>
#include "LinkedList.h"
#include "Element.h"

typedef struct sListIterator {
	LinkedList * list;
	Node * actual;
	Node * anterior;

} ListIterator;

//constructor del iterador
ListIterator * cListIterator(LinkedList * rs);

//1 si hay más elementos 0 sino los hay.
int moreElements(ListIterator * iter);

//obtiene el siguiente elemento no leido de la lista
Element * nextElement(ListIterator * iter);
//obtiene el elemento actual sin avanzar en la lista.
Element * actualElement(ListIterator * iter);
Element * removeElement(ListIterator * iter);
void remove(ListIterator * iter);
//pone el punero actual al primero del resultSet rs
void restart(ListIterator * iter);

//destruye un iterador
void destroyListIterator(ListIterator * iter);

#endif
