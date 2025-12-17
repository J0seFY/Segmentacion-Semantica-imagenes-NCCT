/*
 * LinkedList.h
 *
 *  Created on: 09-10-2011
 *      Author: miguel
 */

#ifndef LINKEDLIST_H_
#define LINKEDLIST_H_

#include <stdlib.h>
#include "Element.h"

/**
 * Este TDA permite la manipulación de listas de elementos.  Es una lista
 * enlazada con recorrido secuencial, inserción secuencial
 *
 */

//estructura para un nodo de la lista enlazada
typedef struct sNode {
	Element * data;
	struct sNode * nextNode;
} Node;

typedef struct sLinkedList {
	Node * raiz;
	Node * ultimo;
	int size;
	int count;//para un conteo del usuario(externo) del tamaño de la estructura
} LinkedList;

//construye un resultset
LinkedList * cLinkedList();

//agrega un elemento al final del ResulSet
void add( LinkedList * list, uint e);
void add(LinkedList * list, Element * x);
void addLast(LinkedList * list, Element * x);
//agrega un elemento al principio
void addFirt(LinkedList * list, Element * x);
//agrega un elemento en orden segun e.key.
void addSort(LinkedList * list, Element * x);
//agrega un elemento en orden segun e.key. si este no se encuentra,
//si este está no lo inserta.
void addSortUniq(LinkedList * list, Element * x);
//elimina el primer elemento
Element * removeFirst(LinkedList * list);
//crea una nueva lista ordenada (los elementos se comparten con list)
LinkedList * listSort(LinkedList * list);
//agrega todos los elementos de fin en la lista ini, eliminandolos de la lista fin
void unir(LinkedList * inicio, LinkedList * fin);
//destruye un Resulset
void destroyLinkedList(LinkedList * list);

//entrega el largo de la lista
int linkedListSize(LinkedList * list);
int empty(LinkedList * list);
//destruye un la lista sin eliminar el elemento contenido por ser necesario en otra estructura
void destroyLinkedListWhithoutElement(LinkedList * list);
void destroyLinkedListWhithoutElementPointer(LinkedList * list);

#endif /* RESULTSET_H_ */
