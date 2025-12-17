#ifndef TEMPORAL_SECUENCE_INCLUDED
#define TEMPORAL_SECUENCE_INCLUDED

//      TDA_TemporalSecuence.h
//      
//      Copyright 2011 Miguel Romero <miguel.romero@ubiobio.cl>
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

#include <time.h>
#include "Element.h"
#include "Instant.h"
#include "LinkedList.h"

//DEFINICIÓN DE LA ESTRUCTURA DE DATO
typedef struct sTDA_TemporalSecuence {
	//Número máximo de elementos
	int N;   //Número máximo de elementos
	int actual;   //número actual de elementos
	Instant startTime;   //instante 0
	int samplingSpeed;   // velocidad de muestreo en segundos
	Element ** data;   // arreglo de punteros a los Elementos
} TemporalSecuence;

//DEFINICIÓN DE SUS OPERACIONES
//constructor de la estructura
TemporalSecuence * cTemporalSecuence(int N, Instant startTime, int sp);

//destructor de la estructura
void destroyTemporalSecuence(TemporalSecuence * ts);
void destroyTemporalSecuence(TemporalSecuence * ts, void destroyElement(Element *));
/*agrega un elemento al final de la secuencia y devuelve 1, 
 * si esta está llena retorna 0;
 */
int add(TemporalSecuence * ts, Element * e);
int copyLast(TemporalSecuence * ts);
//agrega un elemento en la posición x si x < n
int addInTime(TemporalSecuence * ts, int x, Element * e);
/*obtiene el elemento que está en el instante t
 * se considera que hay un elemento por instante dado que se hace un
 * muestreo discreto del mundo mediante snapshot
 */
Element * timeSlice(TemporalSecuence * ts, Instant t);
//retorna el elemenento guardado en la posición i
void * getIth(TemporalSecuence * ts, uint i);

/* Obtiene la lista de elementos que hay entre el instante t1 y el 
 * instante t2.  se debe satisfacer que t1<t2, pues forman un intervalo
 * si t1=t2, entonces no es un intervalo sino un instante.
 */
LinkedList * timeInterval(TemporalSecuence * ts, Instant t1, Instant t2);

//operaciones privadas
int instant2position(TemporalSecuence * ts, Instant t);
//calcula la posición para un instante dado, pero no sobre una
//secuencia sino dado una posición
int instant2pos(Instant start, Instant q, int samplingSpeed);
Instant position2instant(TemporalSecuence * ts, int pos);

size_t sizeTemporalSecuence(TemporalSecuence *ts, size_t sizeElement(Element * e));

#endif
