#ifndef INSTANT_INCLUDED
#define INSTANT_INCLUDED

//      Instant.h
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

#include <time.h>

typedef struct sInstant {
	time_t timeStamp;
} Instant;

//constructores
//crea un instante con la hora actual
Instant cInstant();

//crea un instante a partir de una string con la fecha y hora
//formato largo
Instant cInstant(char * dateTime);

//crea un instante a partir de un time_t
Instant cInstant(time_t t);

//representación en texto del instante
//formato largo
//char * toString(Instant i);

//retorna el time stamp del instante

time_t getTimeStamp(Instant t);

//indica si dos instantes son iguales
int equals(Instant a, Instant b);

/* compara dos instantes y devuelve:
 * un número menor a 0 si a es menor 
 * 0 si son iguales, 
 * un número mayor a 0 si b es mayor
 */
double compareInstant(Instant a, Instant b);

//entrega la diferencia en segundos de dos instantes (valor absoluto)
double diff(Instant a, Instant b);

//suma x segundos al instante i
Instant sum(Instant i, int x);

//como no es un puntero no necesita destructor;

#endif
