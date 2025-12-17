/*
 * InputAdapter.cpp
 *
 *  Created on: 12-10-2011
 *      Author: miguel
 */
#include <stdio.h>
#include <stdlib.h>
#include "InputAdapter.h"

//crea y abre un InputAdapter
InputAdapter * openInputAdapter(const char * origen) {
	InputAdapter * ia = (InputAdapter *) malloc(sizeof(InputAdapter));
	if (ia == NULL) {
		fprintf(stderr, "\n falla malloc en cLinkedList()");
		exit(1);
	}

	ia->arch = fopen(origen, "r");
	if (ia->arch == NULL) {
		fprintf(stderr, "\n falla fopen(%s) en openInputAdapter()", origen);
		exit(1);
	}

	//se leen los metadatos
	int year=2015, month=1, day=1, hh=0, mm=0, ss=0;

//	if(!fscanf(ia->arch, "%d %d %d %d %d %d %d %d \n", &year, &month, &day, &hh,
//			&mm, &ss, &ia->totalSampling, &ia->freqSampling)){
//		fprintf(stderr, "\nerror al leer del archivo de entrada");
//		exit(1);
//	}
	struct tm stm;
	stm.tm_year = year - 1900;
	stm.tm_mon = month - 1;
	stm.tm_mday = day;
	stm.tm_hour = hh;
	stm.tm_min = mm;
	stm.tm_sec = ss;
	ia->startTime = mktime(&stm);
  
  ia->totalSampling=10100;
  //ia->totalSampling=8641;
  //	ia->totalSampling=2592001;
	ia->freqSampling=1;
	return ia;
}

//obtiene el siguiente registro
//PRE: in debe estar abierto y no debe ser fin de archivo
ObjectTimePosition * readInputAdapter(InputAdapter * in) {
	ObjectTimePosition * otp = NULL;
	otp = (ObjectTimePosition *) calloc(1, sizeof(ObjectTimePosition));
	if (otp == NULL) {
		fprintf(stderr, "\n falla calloc en readInputAdapter()");
		exit(1);
	}

	//en el string de fcanf ("%u %u %i %i\n") el \n es importante, pues hace que fscanf lo lea, sino
	//genera un problema y no detecta correctamente el fin de archivo porque queda
	//el último  \n resagado.

	if (!fscanf(in->arch, "%u %u %i %i\n", &(otp->timeStmp), &(otp->idObj),
				&(otp->x), &(otp->y))) {
		free(otp);
		otp = NULL;
	}

//	fprintf(stderr, "Leyendo: IdOBJ: %u, Timestamp: %u, X: %i, Y:%i\n",
//			otp->idObj, otp->timeStmp, otp->x, otp->y);
	if(otp->x<0 || otp->y <0){
		fprintf(stderr, "Error: no se pueden procesar posiciones negativas");
exit(1);
	}
	return otp;
}

//verdadero (1) si hay más elementos, falso (0) en caso contrario
int moreElement(InputAdapter * in) {
	/*
	 * hay más elementos si no se ha llegado al final del archivo
	 *
	 */
	return !feof(in->arch);
}

//cierra el InputAdapter in (1 si lo cierra 0 si no lo hace)
int closeInputAdapter(InputAdapter * in) {
	int resp = fclose(in->arch);
	//fclose entrega 0 si cierra correctamente y EOF en caso contrario.
	resp = (resp == 0) ? 1 : 0;
	return resp;
}
