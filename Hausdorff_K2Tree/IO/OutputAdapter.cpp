/*
 * OutputAdapter.cpp
 *
 *  Created on: 21-10-2011
 *      Author: miguel
 */

#include "OutputAdapter.h"

//crea y abre un OutputAdapter
OutputAdapter * openOutputAdapter(const char * origen) {
	OutputAdapter * oa = (OutputAdapter *) malloc(sizeof(OutputAdapter));
	oa->arch = fopen(origen, "w");
	return oa;

}

//Escribe el objeto en el OutputAdapter
//no muestro el timeStamp por ahora...
void writeOutputAdapter(OutputAdapter * out, ObjectTimePosition * otp) {
	if ((otp->timeStmp == 99999) && (otp->x = 99999) && (otp->y ==-99999)) {
		fprintf(out->arch, "%u\n", otp->idObj);
	} else {
		fprintf(out->arch, "%u %u %i %i\n", otp->timeStmp, otp->idObj, otp->x,
				otp->y);
	}

}

//Escribe el objeto en el OutputAdapter
void writeComment(OutputAdapter * out, const char * coment) {
	fprintf(out->arch, "%s\n", coment);
}

//cierra el InputAdapter in
int closeOutputAdapter(OutputAdapter * out) {

	int resp = fclose(out->arch);
	//fclose entrega 0 si cierra correctamente y EOF en caso contrario.
	resp = (resp == 0) ? 1 : 0;
	return resp;
}

