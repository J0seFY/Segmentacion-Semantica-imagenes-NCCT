//      Element.c
//      
//      Copyright 2011 Miguel Romero <miguel@miguel-laptop>
//      
//      This program is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//      
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//      
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//      MA 02110-1301, USA.

#include "Element.h"
#include <stdlib.h>
#include <stdio.h>

static long int contNew = 0;
static long int contFree = 0;

//constructores de la estructura
Element * cElementEmpty() {
	Element * e = (Element *) malloc(sizeof(Element));
	if (e == NULL) {
		fprintf(stderr, "\n falla malloc en cElementEmpty()");
		exit(1);
	}

	e->pt = NULL;
	e->key = 0;
	contNew++;
	return e;
}

//crea un elemento y le asigna el dato pt
Element * cElement(void * pt) {
	Element * e = (Element *) malloc(sizeof(Element));
	if (e == NULL) {
		fprintf(stderr, "\n falla malloc en cElement()");
		exit(1);
	}
	e->pt = pt;
	e->key = 0;
	contNew++;
	return e;
}
Element * cElementKey(int key, void * pt) {
	Element * e = (Element *) malloc(sizeof(Element));
	if (e == NULL) {
		fprintf(stderr, "\n falla malloc en cElementKey()");
		exit(1);
	}
	e->pt = pt;
	e->key = key;
	contNew++;
	return e;
}

//destructor del interior de la estructura
// los punteros a Element hay de destruirlos afuera.
void destroy(Element * e) {
	//hay que revisar pt no sea nulo
	if (e != NULL && e->pt != NULL) {
		free(e->pt);   //si no hay cuidado puede ocurrir un doble free
		e->pt = NULL;
		contFree++;
	}
}

//permite obtener un puntero al dato guardado
void * get(Element * e) {
	return e->pt;
}
//obtener la clave
int getKey(Element * e) {
	return e->key;
}
//permite asignar un nuevo elemento
void set(Element * e, void * pt) {
	/*Se elmina el dato anterior, primero, esto para evitar que el 
	 * dato siga ocupando espacio si ya no es accedido por nadie
	 */
	free(e->pt);
	e->pt = pt;
	contNew++;
	contFree++;
}
void setKey(Element * e, int key) {
	e->key = key;
}

long int * estadisticaElement() {
	long int * resp = (long int *) calloc(2, sizeof(long int));
	if (resp == NULL) {
		fprintf(stderr, "\n falla malloc en estadisticaElement()");
		exit(1);
	}
	resp[0] = contNew;
	resp[1] = contFree;
	return resp;
}
