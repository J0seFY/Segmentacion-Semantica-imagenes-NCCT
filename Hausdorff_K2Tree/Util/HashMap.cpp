/*
 * HashMap.cpp
 *
 *  Created on: 20-10-2011
 *      Author: miguel
 */
#include "ListIterator.h"
#include "HashIter.h"
#include "HashMap.h"

int hashFun(HashMap * hm, unsigned int key, int numIntento) {
	return (key + numIntento) % hm->initialCapacity;
	//el numero que divide idealmente es un numero primo
}

//crea un nuevo hashmap
HashMap * cHashMap(int initialCapacity) {
	HashMap * hm = (HashMap *) malloc(sizeof(HashMap));
	if (hm == NULL) {
		fprintf(stderr, "\n falla malloc en cLinkedList()");
		exit(1);
	}

	hm->initialCapacity = (int) (initialCapacity * 1.25);
	hm->map = (Element **) calloc(hm->initialCapacity, sizeof(Element*));
	if (hm->map == NULL) {
		fprintf(stderr, "\n falla malloc en cHashMap()");
		exit(1);
	}

	hm->size = 0;
	hm->loadFactor = 75;
	if (hm->map == NULL) {
		return NULL;
	}
	return hm;
}

//borra todos los elementos del hashmap
HashMap * clearHashMap(HashMap *hm) {
	for (uint i = 0; i < hm->initialCapacity; i++) {
		if (hm->map[i] != NULL) {
			destroy(hm->map[i]);
			hm->map[i] = NULL;
		}
	}
	hm->size = 0;
	return hm;
}

void destroyHashMapWhithoutElement(HashMap *hm) {
	for (uint i = 0; i < hm->initialCapacity; i++) {
		if (hm->map[i] != NULL) {
			//destroy(hm->map[i]);
			hm->map[i] = NULL;
		}
	}
	free(hm->map);
	hm->map = NULL;
	hm->size = 0;
}

void destroyHashMap(HashMap *hm) {
	for (uint i = 0; i < hm->initialCapacity; i++) {
		if (hm->map[i] != NULL) {
			destroy(hm->map[i]);
			free(hm->map[i]);
			hm->map[i] = NULL;
		}
	}
	free(hm->map);
	hm->map = NULL;
	hm->size = 0;
}

//indica el número de elementos almacenados
int sizeOfHashMap(HashMap *hm) {
	return hm->size;
}

//verdadero si todas las claves de la lista estan en el hashmap
int containsAllKey(HashMap * hm, LinkedList * list) {
	int resp = 1;
	ListIterator * iter = cListIterator(list);
	while ((resp == 1) && (moreElements(iter))) {
		resp = containsKey(hm, getKey(nextElement(iter)));
	}
	return resp;
}
//verdadero si está el elemento de clave key o falso sino lo esta
int containsKey(HashMap * hm, int key) {
	uint cont = 0;
	int pos = 0;
	do {
		pos = hashFun(hm, key, cont);
		cont++;
	} while ((hm->map[pos] != NULL) && (getKey(hm->map[pos]) != key)
			&& (cont < hm->size));

	if ((hm->map[pos] != NULL) && (getKey(hm->map[pos]) == key)) {
		return 1;
	} else {
		return 0;
	}
}
//obtienen un elemento a partir de su clave, si no está da null
Element * get(HashMap * hm, int key) {
	uint cont = 0;
	int pos = 0;
	do {
		pos = hashFun(hm, key, cont);
		cont++;
	} while ((hm->map[pos] != NULL) && (getKey(hm->map[pos]) != key)
			&& (cont < hm->size));

	if ((hm->map[pos] != NULL) && (getKey(hm->map[pos]) == key)) {
		return hm->map[pos];
	} else {
		return NULL;
	}
}
//elimina un elemento del hashmap si este se encuentra retorna el objeto,sino null.
Element * remove(HashMap *hm, int key) {
	uint cont = 0;
	int pos = 0;
	do {
		pos = hashFun(hm, key, cont);
		cont++;
	} while ((hm->map[pos] != NULL) && (getKey(hm->map[pos]) != key)
			&& (cont < hm->size));

	if ((hm->map[pos] != NULL) && (getKey(hm->map[pos]) == key)) {
		Element * e = hm->map[pos];
		hm->map[pos] = NULL;
		hm->size--;
		return e;
	} else {
		return NULL;
	}

}
//triplica la capacidad del hashmap.
HashMap * resize(HashMap * hm) {
	fprintf(stderr,"Cuidado, Se ha redimensionado el hashmap, esto afecta en los tiempos de los experimentos");
	HashMap * newHashMap = cHashMap(hm->initialCapacity * 3);
	if (newHashMap == NULL)
		return NULL;
	for (uint i = 0; i < hm->initialCapacity; i++) {
		if (hm->map[i] != NULL) {
			put(newHashMap, hm->map[i]);
		}
	}
	hm->initialCapacity = newHashMap->initialCapacity;
	hm->map = newHashMap->map;
	newHashMap->map = NULL;
	hm->size = newHashMap->size;
	return hm;
}

//agrega un elemento a un HashMap si no está. si el elemento está lo actualiza.
Element * put(HashMap *hm, Element * e) {
	uint cont = 0;
	int pos = 0;
	int key = e->key;
	do {
		pos = hashFun(hm, key, cont);
		cont++;
	} while ((hm->map[pos] != NULL) && (getKey(hm->map[pos]) != key)
			&& (cont < hm->size));

	if ((hm->map[pos] == NULL)) {
		//el elemento no está hay que insertarlo
		hm->map[pos] = e;
		hm->size++;
		return e;
	} else if (getKey(hm->map[pos]) == key) {
		//El elemento está en el hashMap, hay que actualizarlo
		destroy(hm->map[pos]);
		free(hm->map[pos]);
		hm->map[pos] = e;
		return e;
	} else {
		fprintf(stderr, "Se ha llenado el hashmap!!!");
		exit(1);
		//si el contenedor esta lleno lo amplifico
		if (resize(hm)) {
			put(hm, e);
			return e;
		} else {
			//si no se pudo aumentar el tamaño
			return NULL;
		}

	}

}
//simplifica la creación de una tabla hash de enteros
Element * put(HashMap *hm, uint e) {
	uint * num = (uint *) malloc(sizeof(uint));
	*num = e;
	Element * elem = cElementKey(e, num);
	return (put(hm, elem));
}
//añade todos los elementos de la lista list
void putAll(HashMap *hm, LinkedList * list) {

	ListIterator * iter = cListIterator(list);

	while (moreElements(iter)) {
		Element * e = nextElement(iter);
		//pongo el elemento en el HashMap, si esta lleno, entonces envío un mensaje de error.
		if (put(hm, e) == NULL) {
			//no es posible insertar un elemento en el hashmap
			perror("Warning: No fue posible insertar el elemento en el hashmap");
			exit(1);
		}
	}

}
//retorna una lista con todos los elementos
LinkedList * getAll(HashMap * hm) {
	LinkedList * resp = cLinkedList();

	for (uint i = 0; i < hm->initialCapacity; i++) {
		if (hm->map[i] != NULL) {
			add(resp, hm->map[i]);
		}
	}

	return resp;
}
//retorna una lista con todos los elementos ordenados por Key
LinkedList * getAllSort(HashMap * hm) {
	LinkedList * resp = cLinkedList();

	for (uint i = 0; i < hm->initialCapacity; i++) {
		if (hm->map[i] != NULL) {
			addSort(resp, hm->map[i]);
		}
	}

	return resp;
}
//dos hashmap son iguales si tienen el mismo número de elementos y
//si las claves de los elementos contenidos en iguales.
//no discrimina si dos objetos son realmente iguales, solo si coincide su OID.

int equals(HashMap *a, HashMap *b) {
	if (a->size != b->size) {
		//si no tienen el mismo número de elementos no son iguales.
		return 0;
	} else {
		//si tienen el mismo número de elmentos, todos los elementos de b
		//deben estar en a para que sean iguales.
		HashIterator * iter = cHashIterator(b);
		Element * e;
		while (moreElements(iter)) {
			e = nextElement(iter);
			if (!containsKey(a, e->key)) {
				return 0;
			}
		}
	}
	return 1;
}
//intersección de conjuntos sobre dos hashMap, el resultado es un tercero,
//no altera los dos primeros.

