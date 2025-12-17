/*
 * ResultSet.c
 *
 *  Created on: 09-10-2011
 *      Author: miguel
 */

#include <stdlib.h>
#include <stdio.h>
#include "LinkedList.h"

//construye una ista vacía
LinkedList * cLinkedList() {
	LinkedList * rs = (LinkedList *) malloc(sizeof(LinkedList));
	if (rs == NULL) {
		fprintf(stderr, "\n falla malloc en cLinkedList()");
		exit(1);
	}
	rs->raiz = NULL;
	rs->size = 0;
	rs->count = 0;
	rs->ultimo = NULL;
	return rs;
}
//simplifica la creación de una lista enlazada de enteros
void add(LinkedList * list, uint e) {
	uint * num = (uint *) malloc(sizeof(uint));
	*num = e;
	Element * elem = cElementKey(e, num);
	return (add(list, elem));
}

//agrega un elemento al final de la lista
void add(LinkedList * list, Element * x) {

	//PRE: list no es nulo
	if (list == NULL) {
		return;
	}

	//creo un nuevo nodo que contenga a x,
	Node * newNode = (Node *) malloc(sizeof(Node));
	if (newNode == NULL) {
		fprintf(stderr, "\n falla malloc en add() (List)");
		exit(1);
	}

	newNode->data = x;
	newNode->nextNode = NULL;   //no hay siguiente pues es el último

	//si es el primer elemento es un caso especial
	if (list->size == 0) {
		list->raiz = newNode;
		list->ultimo = list->raiz;
	} else {
		//lo agrego al final y actualizo el último
		list->ultimo->nextNode = newNode;
		list->ultimo = newNode;
	}
	list->size++;   //incremento el contador de nodos;
}
void addLast(LinkedList * list, Element * x) {
	add(list, x);
}
//agrega un elemento al principio
void addFirt(LinkedList * list, Element * x) {
	//PRE: list no es nulo
	if (list == NULL) {
		return;
	}

	//creo un nuevo nodo que contenga a x,
	Node * newNode = (Node *) malloc(sizeof(Node));
	if (newNode == NULL) {
		fprintf(stderr, "\n falla malloc en addFirst()");
		exit(1);
	}

	newNode->data = x;
	newNode->nextNode = NULL;   //no hay siguiente pues es el último

	//si es el primer elemento es un caso especial
	if (list->size == 0) {
		list->raiz = newNode;
		list->ultimo = list->raiz;
	} else {
		//lo agrego al principio y actualizo la raiz
		newNode->nextNode = list->raiz;
		list->raiz = newNode;
	}
	list->size++;   //incremento el contador de nodos;

}
//agrega un elemento en orden segun e.key.
void addSort(LinkedList * list, Element * x) {
	//PRE: list no es nulo
	if (list == NULL) {
		return;
	}

	//creo un nuevo nodo que contenga a x,
	Node * newNode = (Node *) malloc(sizeof(Node));
	if (newNode == NULL) {
		fprintf(stderr, "\n falla malloc en addSort()");
		exit(1);
	}

	newNode->data = x;
	newNode->nextNode = NULL;   //no hay siguiente pues es el último

	//si es el primer elemento es un caso especial
	if (list->size == 0) {
		list->raiz = newNode;
		list->ultimo = list->raiz;
	} else {
		//lo agrego en la posición que corresponda
		Node * iter = list->raiz;
		Node * ant = iter;
		//busco la posición a insertar el nodo
		while ((iter != NULL) && (iter->data->key < newNode->data->key)) {
			ant = iter;
			iter = iter->nextNode;
		}
		if (iter == list->raiz) {
			//se inserta al inicio (equivalente a addFirst(list,x);)
			newNode->nextNode = list->raiz;
			list->raiz = newNode;
		} else if (iter == NULL) {
			//se inserta al último (equivalente a add(list,x);)
			list->ultimo->nextNode = newNode;
			list->ultimo = newNode;
		} else {
			//inserta en un lugar intermedio
			newNode->nextNode = iter;
			ant->nextNode = newNode;
		}

	}
	list->size++;   //incremento el contador de nodos;

}
//agrega un elemento en orden segun e.key. si este no se encuentra,
//si este está no lo inserta.
void addSortUniq(LinkedList * list, Element * x) {
	//PRE: list no es nulo
	if (list == NULL) {
		return;
	}

	//creo un nuevo nodo que contenga a x,
	Node * newNode = (Node *) malloc(sizeof(Node));
	if (newNode == NULL) {
		fprintf(stderr, "\n falla malloc en addSort()");
		exit(1);
	}

	newNode->data = x;
	newNode->nextNode = NULL;   //no hay siguiente pues es el último

	//si es el primer elemento es un caso especial
	if (list->size == 0) {
		list->raiz = newNode;
		list->ultimo = list->raiz;
	} else {
		//lo agrego en la posición que corresponda
		Node * iter = list->raiz;
		Node * ant = iter;
		//busco la posición a insertar el nodo
		while ((iter != NULL) && (iter->data->key < newNode->data->key)) {
			ant = iter;
			iter = iter->nextNode;
		}
		if (iter == list->raiz && (iter->data->key != newNode->data->key)) {
			//se inserta al inicio (equivalente a addFirst(list,x);)
			newNode->nextNode = list->raiz;
			list->raiz = newNode;
			list->size++;
		} else if (iter == NULL) {
			//se inserta al último (equivalente a add(list,x);)
			list->ultimo->nextNode = newNode;
			list->ultimo = newNode;
			list->size++;
		} else if (iter->data->key != newNode->data->key) {
			//inserta en un lugar intermedio
			newNode->nextNode = iter;
			ant->nextNode = newNode;
			list->size++;
		}
	}
}

//elimina el primer elemento
Element * removeFirst(LinkedList * list) {
	Element * x;
	Node * p;
	if (list == NULL || list->size == 0) {
		return NULL;
	}
	x = list->raiz->data;
	p = list->raiz;
	list->raiz = list->raiz->nextNode;
	if (list->size == 1) {
		list->ultimo = NULL;
	}
	p->nextNode = NULL;
	free(p);
	p = NULL;
	list->size--;
	return x;
}

//crea una nueva lista ordenada mediante insertSort, peor caso N^2.
//se comparten los elementos con la lista anterior.
LinkedList * listSort(LinkedList * list) {
	LinkedList * resp = cLinkedList();
	if (empty(list)) {
		return resp;
	}
	Node * iter = list->raiz;
	while (iter != NULL) {
		addSort(resp, iter->data);
		iter = iter->nextNode;
	}
	return resp;
}

int empty(LinkedList * list) {
	return ((list == NULL) || (list->size == 0)) ? 1 : 0;
}

/*
 * engancha a la lista inicio la lista fin.  hay que tener cuidado
 * pues no genera una nueva lista con la unión de ambos,
 * sino que engancha a la primera la segunda y luego
 * por seguridad deja la segunda lista vacía.
 * pre: ambas no son nulas...
 */
void unir(LinkedList * inicio, LinkedList * fin) {
	if (empty(fin)) {
		return;
	} else if (empty(inicio)) {
		inicio->size = fin->size;
		inicio->raiz = fin->raiz;
		inicio->ultimo = fin->ultimo;
	} else {
		inicio->size += fin->size;
		inicio->ultimo->nextNode = fin->raiz;
		inicio->ultimo = fin->ultimo;

	}
	//vacío la segunda lista por seguridad
	//conceptualmente los elementos de fin se ha traspasado al final de ini.
	fin->size = 0;
	fin->raiz = NULL;
	fin->ultimo = NULL;
}

//destruye un Resulset
void destroyLinkedList(LinkedList * list) {
	if (list == NULL) {
		return;
	}
	Node * aux;
	int i;
	//elimino todos los elementos de la lista
	for (i = 0; i < (list->size); ++i) {
		aux = list->raiz;
		list->raiz = aux->nextNode;   //avanza la raiz.
		aux->nextNode = NULL;   //desengancho el nodo a eliminar.
		//lieberación del contenido del nodo
		destroy(aux->data);
		free(aux->data);
		aux->data = NULL;
		//liberación del nodo.
		free(aux);
		aux = NULL;
	}
	aux = NULL;
	//pongo en null los punteros de list
	list->raiz = NULL;
	list->ultimo = NULL;
	list->size = 0;
}
//destruye un la lista sin eliminar el elemento contenido por ser necesario en otra estructura
void destroyLinkedListWhithoutElement(LinkedList * list) {
	Node * aux;
	int i;
	//elimino todos los elementos de la lista
	for (i = 0; i < (list->size); ++i) {
		aux = list->raiz;
		list->raiz = aux->nextNode;   //avanza la raiz.
		aux->nextNode = NULL;   //desengancho el nodo a eliminar.
		//NO se libera el elemento
		aux->data = NULL;
		//liberación del nodo.
		free(aux);
		aux = NULL;
	}
	aux = NULL;
	//pongo en null los punteros de list
	list->raiz = NULL;
	list->ultimo = NULL;
	list->size = 0;
}
//destruye un la lista sin eliminar el puntero dentro del elemento contenido por ser necesario en otra estructura
void destroyLinkedListWhithoutElementPointer(LinkedList * list) {
	Node * aux;
	int i;
	//elimino todos los elementos de la lista
	for (i = 0; i < (list->size); ++i) {
		aux = list->raiz;
		list->raiz = aux->nextNode;   //avanza la raiz.
		aux->nextNode = NULL;   //desengancho el nodo a eliminar.
		//se libera el elemento pero no a lo que apunta.
		aux->data->pt=NULL;
		free(aux->data);
		aux->data = NULL;
		//liberación del nodo.
		free(aux);
		aux = NULL;
	}
	aux = NULL;
	//pongo en null los punteros de list
	list->raiz = NULL;
	list->ultimo = NULL;
	list->size = 0;
}
int linkedListSize(LinkedList * list) {
	return (list->size);
}
