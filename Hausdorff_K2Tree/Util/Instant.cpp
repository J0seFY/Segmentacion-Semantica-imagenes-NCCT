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

#include "Instant.h"
#include <time.h>

//typedef struct Instant{
//time_t timeStamp;
//}

//constructores
//crea un instante con la hora actual
Instant cInstant() {
	Instant i;
	time(&i.timeStamp);
	return i;
}

//crea un instante a partir de una string con la fecha y hora
//TODO: falta por desarrollar
//dd/mm/YYYY HH:MM:SS
Instant cInstant(char * dateTime) {
	Instant i;
	struct tm t;

	//t= función para pasar de string a struct tm;
	i.timeStamp = mktime(&t);
	return i;

}

//crea un instante a partir de un time_t
Instant cInstant(time_t t) {
	Instant i;
	i.timeStamp = t;
	return i;
}

/*representación en texto del instante en formato:
 * 		Www Mmm dd hh:mm:ss yyyy 
 *Where:
 * 		Www is the weekday, 
 * 		Mmm the month in letters, 
 * 		dd the day of the month, 
 * 		hh:mm:ss the time, 
 * 		and yyyy the year.
 * 
 */

//char * toString(Instant i) {
//todo: falta por implementar
//}
time_t getTimeStamp(Instant t) {
	return t.timeStamp;
}

//indica 1 si dos instantes son iguales o 0 si no lo son
int equals(Instant a, Instant b) {
	int resp;
	if (0 == (a.timeStamp - b.timeStamp)) {
		resp = 1;
	} else {
		resp = 0;
	}
	return (resp);
}

/* compara dos instantes y devuelve:
 * un número menor a 0 si a es menor 
 * 0 si son iguales, 
 * un número mayor a 0 si b es mayor
 */
double compareInstant(Instant a, Instant b) {
	return (a.timeStamp - b.timeStamp);
}

//entrega la diferencia en segundos de dos instantes
double diff(Instant a, Instant b) {
	return (difftime(a.timeStamp, b.timeStamp));
}

//suma x segundos al instante i
Instant sum(Instant i, int x) {
	return (cInstant(i.timeStamp + x));
}

//como no hay punteros no necesita destructor;

