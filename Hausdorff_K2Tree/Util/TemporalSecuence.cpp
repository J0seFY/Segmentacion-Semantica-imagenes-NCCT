//      TDA_TemporalSecuence.c
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

#include <stdio.h>
#include <stdlib.h>
#include "TemporalSecuence.h"
#include "Instant.h"
#include "LinkedList.h"
#include "ListIterator.h"
#include "Element.h"

//constructor de la estructura
TemporalSecuence * cTemporalSecuence(int N, Instant startTime, int sp) {
	TemporalSecuence * ts = (TemporalSecuence *) malloc(
			sizeof(TemporalSecuence));
	if (ts == NULL) {
		fprintf(stderr, "\n falla malloc ts");
		exit(1);
	}
	ts->N = N;
	ts->actual = 0;
	ts->startTime = startTime;
	ts->samplingSpeed = sp;
	ts->data = (Element **) calloc(N, sizeof(Element*));
	if (ts->data == NULL) {
		fprintf(stderr, "\n falla calloc ts->data");
		exit(1);
	}
	return ts;
}

//destructor de la estructura (solo el interior no el puntero externo)
void destroyTemporalSecuence(TemporalSecuence * ts) {
	//destruir el interior del arreglo data
	for (int i = 0; i < ts->actual; i++) {
		//	printf("destroy(data[%i])\n", i);
		destroy(ts->data[i]);
		//	printf("free(data[%i])\n", i);
		free(ts->data[i]);
		ts->data[i] = NULL;
	}
	free(ts->data);
	ts->data = NULL;
	ts->N = 0;
	ts->actual = 0;
	//ts->startTime=0;
	ts->samplingSpeed = 0;
}

//destructor de la estructura
void destroyTemporalSecuence(TemporalSecuence * ts, void destroyElement(Element *)) {
	//destruir el interior del arreglo data
	for (int i = 0; i < ts->actual; i++) {
		//	printf("destroy(data[%i])\n", i);
		destroyElement(ts->data[i]);
		//	printf("free(data[%i])\n", i);
		free(ts->data[i]);
		ts->data[i] = NULL;
	}
	free(ts->data);
	ts->data = NULL;
	ts->N = 0;
	ts->actual = 0;
	//ts->startTime=0;
	ts->samplingSpeed = 0;
}
//agrega un elemento al final de la secuencia, si esta está llena retorna 0;
int add(TemporalSecuence * ts, Element * e) {
	if ((ts != NULL) && ((ts->actual) < (ts->N))) {
		ts->data[ts->actual++] = e;
		return 1;
	}
	return 0;
}
//se copia el último elemento de la secuencia
int copyLast(TemporalSecuence * ts){
	if ((ts != NULL) && ((ts->actual) < (ts->N))) {
			ts->data[ts->actual] = ts->data[ts->actual-1];
			ts->actual++;
			return 1;
		}
		return 0;
}
//agrega un elemento en el instante x y ajusta el puntero actual
int addInTime(TemporalSecuence * ts, int x, Element * e) {
	if ((ts != NULL) && ((ts->actual) < (ts->N)) && (x < (ts->N))) {
		ts->data[x] = e;
		ts->actual = ++x;   //actual debe apunta siempre uno despúes del último
		return 1;
	}
	return 0;
}

/*obtiene el elemento que está en el instante t
 * se considera que hay un elemento por instante dado que se hace un
 * muestreo discreto del mundo mediante snapshot retorna un puntero
 * al elemento de la estructura, lo que permite modificar el contenido
 * de dicho elemento, esto hay que revisar si es lo adecuado.
 */
Element * timeSlice(TemporalSecuence * ts, Instant t) {
	/* el instante debe formar parte del arreglo, 
	 * sino el resultado es NULL
	 */

	if ((compareInstant(t, ts->startTime) >= 0)
			&& (compareInstant(t, position2instant(ts, ts->actual)) <= 0)) {
		return (ts->data[instant2position(ts, t)]);
	} else {
		return NULL;
	}
}

/* ****
 * Obtiene la lista de elementos que hay entre el instante t1 y el
 * instante t2.  se debe satisfacer que t1<t2, pues forman un intervalo
 * si t1=t2, entonces no es un intervalo sino un instante.
 *
 */
LinkedList * timeInterval(TemporalSecuence * ts, Instant t1, Instant t2) {

	/* con el fin de cumplir la precondición de la función 
	 * instant2position que requiere que el instante ocurra dentro del
	 * arry se deben evaluar los siguientes casos 3 casos:
	 * 
	 *  
	 * si el intervalo a consultar[t1..t2] ocurre antes o despúes que 
	 * el intervalo formado [ts->startTime..ts->position2Interval(actual-1)]
	 * el resultado es NULL.
	 */

	if ((compareInstant(t2, ts->startTime) < 0)
			|| (compareInstant(t1, position2instant(ts, ts->actual - 1)) > 0)) {
		return NULL;
	}
	/* tambien puede ocurrir que el intervalo a consultar comience antes
	 * del startTime, pero termine dentro de ts. en este caso se entrega
	 * como resultado lo que se conoce que es desde startTime hasta t2
	 * 
	 */
	if ((compareInstant(t1, ts->startTime) < 0)
			&& (compareInstant(t2, position2instant(ts, ts->actual - 1)) <= 0)) {
		t1 = ts->startTime;
	}
	/* otro caso.  
	 * Cuando t1 ocurre después de startTime (y antes que ts->actual)
	 * y t2 ocurre despúes que ts->actual. Aqui tambien se puede 
	 * responder pero solo la porción conocida.
	 */
	if ((compareInstant(t1, ts->startTime) >= 0)
			&& (compareInstant(t2, position2instant(ts, ts->actual - 1)) >= 0)) {
		t2 = position2instant(ts, ts->actual - 1);
	}

	//calcular la cantidad de elementos que hay entre t1 y t2
	int fin = instant2position(ts, t2);
	int ini = instant2position(ts, t1);
	//creo una lista para la salida
	LinkedList * list = cLinkedList();
	//asigno los elementos que correspondan a la lista de salida
	for (register int i = ini; i <= fin; i++) {
		add(list, (ts->data[i]));
	}
	return list;
}

//dado un instante entrega la posición correspondiente de la secuencia
//PRE: t es un instante que está dentro del arreglo [0..N-1]
int instant2position(TemporalSecuence * ts, Instant t) {
	double segundos = diff(t, ts->startTime);
	double pos = (segundos / ts->samplingSpeed);
	return ((int) pos);

}
//calcula la posición para un instante dado, pero no sobre una
//secuencia sino dado una posición
int instant2pos(Instant start, Instant q, int samplingSpeed){
	double segundos = diff(q, start);
	double pos = (segundos / samplingSpeed);
	return ((int) pos);

}
//dada una posición de la secuencia entrega el instante correspondiente
Instant position2instant(TemporalSecuence * ts, int pos) {
	return (sum(ts->startTime, (pos * ts->samplingSpeed)));
}

size_t sizeTemporalSecuence(TemporalSecuence *ts,
							size_t sizeElement(Element * e)) {
	if (ts == NULL) {
		return 0;
	}
	size_t total = sizeof(TemporalSecuence);
	for (register int i = 0; i < ts->actual; i++) {
		total += sizeElement(ts->data[i]);
	}
	return total;
}

void * getIth(TemporalSecuence * ts, uint i){
	return ts->data[i]->pt;
}
