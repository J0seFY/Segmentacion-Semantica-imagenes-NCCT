/*
 * ListIterator.cpp
 *
 *  Created on: 10-10-2011
 *      Author: miguel
 */
#include "ListIterator.h"
#include "LinkedList.h"
#include "Element.h"

//constructor del iterador a partir de la lista a iterar
ListIterator * cListIterator(LinkedList * list) {
	if (list == NULL)
		return NULL;

	ListIterator * iter = (ListIterator *) malloc(sizeof(ListIterator));
	iter->list = list;
	iter->actual = list->raiz;
	iter->anterior = NULL;
	return iter;
}

//1 si hay más elementos 0 sino los hay.
int moreElements(ListIterator * iter) {
	//PRE: iter no es nulo
	if ((iter == NULL) || (iter->actual == NULL)) {
		return 0;
	} else {
		return 1;
	}

}

//obtiene el siguiente elemento no leido de la lista
Element * nextElement(ListIterator * iter) {
	//PRE: iter no es nulo ni su apuntador actual

	Element * e = NULL;   //un elemento vacío
	if ((iter == NULL) || (iter->actual == NULL)) {
		return e;
	}
	//obtengo el dato del nodo actual
	e = iter->actual->data;
	iter->anterior = iter->actual;
	//muevo el puntero actual al siguiente
	iter->actual = iter->actual->nextNode;
	return e;
}
//obtiene el elemento actual sin mover el puntero al siguiente
Element * actualElement(ListIterator * iter) {
	//PRE: iter no es nulo ni su apuntador actual

	Element * e = NULL;   //un elemento vacío
	if ((iter == NULL) || (iter->actual == NULL)) {
		return e;
	}
	//obtengo el dato del nodo actual
	e = iter->actual->data;
	return e;
}
//Elimina el elemento actual de la lista, y avanza al siguiente de la lista
//pero no lo elimina de la memoria
//lo retorna al usuario el cual debe elminarlo posteriormente.  Esto para
//evitar el doble free cuando se comparten elementos.
Element * removeElement(ListIterator * iter) {
	Element * e = NULL;   //un elemento vacío
	if ((iter == NULL) || (iter->actual == NULL)) {
		return e;
	}
	//obtengo el dato del nodo actual
	e = iter->actual->data;
	if (iter->anterior == NULL) {
		//caso especial de eliminar la raiz.
		iter->list->raiz=iter->list->raiz->nextNode;
		iter->actual->data=NULL;
		iter->actual->nextNode=NULL;
		free(iter->actual);
		iter->actual=iter->list->raiz;
	} else {
		iter->actual->data = NULL;
		iter->anterior->nextNode = iter->actual->nextNode;
		iter->actual->nextNode = NULL;
		free(iter->actual);
		iter->actual = iter->anterior->nextNode;
	}
	iter->list->size--;
	return e;
}
//Elimina el nodo de la lista y tambien el elemento contenido.
void remove(ListIterator * iter){
	destroy(removeElement(iter));
}
//pone el puntero actual al primero de la lista
void restart(ListIterator * iter) {
	//pre: iter no nulo
	if (iter == NULL) {
		//no se puede hacer nada
	} else {
		iter->actual = iter->list->raiz;
		iter->anterior = NULL;
	}
}

//destruye un iterador
void destroyListIterator(ListIterator * iter) {
	//no hay que destruir la lista original por tanto
	//basta con poner a nul los punteros
	iter->actual = NULL;
	iter->list = NULL;
}
