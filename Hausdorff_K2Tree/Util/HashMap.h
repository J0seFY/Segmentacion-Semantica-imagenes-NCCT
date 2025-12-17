/*
 * HashMap.h
 *
 *  Created on: 20-10-2011
 *      Author: miguel
 */

#ifndef HASHMAP_H_
#define HASHMAP_H_

#include "Element.h"
#include "LinkedList.h"
#include <stdlib.h>
#include <stdio.h>
/*
 * PRE: Cada elemento del HashMap posee una clave única.
 * 		no se aceptan elementos con claves repetidas.
 * 		Este hashmap no se ha implementado completamente,
 * 		solo lo necesario para el problema.
 */
typedef struct sHashMap {
	uint initialCapacity;
	uint size;
	int loadFactor;
	Element ** map;
} HashMap;

//crea un nuevo hashmap
HashMap * cHashMap(int initialCapacity);
void destroyHashMap(HashMap *hm);
void destroyHashMapWhithoutElement(HashMap *hm);
//borra todos los elementos del hashmap
HashMap * clearHashMap(HashMap *hm);
//indica el número de elementos almacenados
int sizeOfHashMap(HashMap *hm);
//verdadero si todas las claves de la lista estan en el hashmap
int containsAllKey(HashMap * hm,LinkedList * list);
//verdadero si está el elemento de clave key o falso sino lo esta
int containsKey(HashMap * hm, int key);
//obtienen un elemento a partir de su clave, si no está da null
Element * get(HashMap * hm, int key);
//elimina un elemento del hashmap si este se encuentra retorna el objeto,sino null.
Element * remove(HashMap *hm, int key);
//agrega un elemento a un HashMap si no está. si el elemento está lo actualiza.
//retorna NULL si esta lleno el hm sino el elemento insertado
Element * put(HashMap *hm, Element * e);
Element * put(HashMap *hm, uint e);
//añade todos los elementos de la lista list
void putAll(HashMap *hm, LinkedList * list);
//retorna una lista con todos los elementos
LinkedList * getAll(HashMap * hm);
//retorna una lista con todos los elementos ordenados por Key
LinkedList * getAllSort(HashMap * hm);

//dos hashmap son iguales si tienen el mismo número de elementos y
//si las claves de los elementos contenidos en iguales.
//no discrimina si dos objetos son realmente iguales, solo si coincide su OID.

int equals(HashMap *a, HashMap *b);


#endif /* HASHMAP_H_ */
