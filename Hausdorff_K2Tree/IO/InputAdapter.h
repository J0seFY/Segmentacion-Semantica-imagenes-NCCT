/*
 * InputAdapter.h
 *
 *  Este TDA permite minimizar el acoplamiento entre la fuente de origen
 *  de los datos y la aplicación.
 *  esta versión lee de un archivo de texto.
 *  Created on: 12-10-2011
 *      Author: miguel
 */

#ifndef INPUTADAPTER_H_
#define INPUTADAPTER_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../Util/ObjectTimePosition.h"


typedef struct sInputAdapter {
	time_t startTime;
	int totalSampling;
	int freqSampling;
	FILE * arch;
} InputAdapter;

//crea y abre un InputAdapter
InputAdapter * openInputAdapter(const char * origen);

//obtiene el siguiente registro
ObjectTimePosition * readInputAdapter(InputAdapter * in);

//verdadero (1) si hay más elementos, falso (0) en caso contrario
int moreElement(InputAdapter * in);

//cierra el InputAdapter in
int closeInputAdapter(InputAdapter * in);

#endif /* INPUTADAPTER_H_ */
