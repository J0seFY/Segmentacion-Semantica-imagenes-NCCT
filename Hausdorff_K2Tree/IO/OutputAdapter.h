/*
 * OutputAdapter.h
 *
 *  Created on: 21-10-2011
 *      Author: miguel
 */

#ifndef OUTPUTADAPTER_H_
#define OUTPUTADAPTER_H_
#include "InputAdapter.h"
#include <stdio.h>

typedef struct sOutputAdapter {
//	time_t startTime;
//	int totalSampling;
//	int freqSampling;
	FILE * arch;
} OutputAdapter;
//crea y abre un InputAdapter
OutputAdapter * openOutputAdapter(const char * origen);

//Escribe el objeto en el OutputAdapter
void writeOutputAdapter(OutputAdapter * out, ObjectTimePosition * otp);
void writeComment(OutputAdapter * out, const char * coment);
//cierra el InputAdapter in
int closeOutputAdapter(OutputAdapter * out);

#endif /* OUTPUTADAPTER_H_ */
