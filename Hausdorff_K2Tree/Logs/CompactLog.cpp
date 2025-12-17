/*
 *
 *
 *  Created on: 17-09-2013
 *      Author: miguel
 */

#include "CompactLog.h"
#include "./gamma.h"
#include "../Util/Factory.h"
#include <ostream>
#include <stdio.h>
void printBitString3(FILE * out, BitSequence * a) {
	fprintf(out, "\n");
	for (size_t i = 0; i < a->getLength(); i++) {
		fprintf(out, "%d", a->access(i));
		if ((i + 1) % 100 == 0) {
			fprintf(out, "\n");
		} else if ((i + 1) % 4 == 0) {
			fprintf(out, " ");
		}
	}
	fprintf(out, "\n");
}

CompactLog::CompactLog(LogRP * logRP) {
	size_t lenLog = logRP->getLenLog();
	//preparar las secuencias
	if ((logRP != NULL) && (lenLog > 1)) {
		ZigZagCoder coder;
		int bitX = 0, bitY = 0;
		//las secuencias comprimidas siempre tienen 0,0 en el tiempo 0.
		//Por lo tanto no es necesario almacenarlo
		int * x = new int[lenLog - 1]();
		int * y = new int[lenLog - 1]();
		ListIterator * iter = logRP->getChangeIter();
		int * act = NULL;
		Element * e = removeElement(iter);
		int * ant = (int *) get(e);
		e->pt = NULL;
		free(e);
		e = NULL;
		uint i = 1;
		while (moreElements(iter)) {
			e = removeElement(iter);
			act = (int *) get(e);
			e->pt = NULL;
			free(e);
			e = NULL;
			x[i - 1] = coder.cod(act[0] - ant[0]);
			y[i - 1] = coder.cod(act[1] - ant[1]);
			//printf("encode: delta(%d, %d), Elias(%d, %d)\n", act[0] - ant[0],act[1] - ant[1],x[i-1],y[i-1]);
			//el número de bits utilizado por un código de elias es=2*floor(log_2(x))+1
			bitX += 2 * floor(log2(x[i - 1] + 1.0)) + 1;
			bitY += 2 * floor(log2(y[i - 1] + 1.0)) + 1;
			i++;
			free(ant);
			ant = act;
		}
		free(ant);
		ant = NULL;
		destroyListIterator(iter);
		free(iter);
		iter = NULL;
		//TODO: aquí se podría aplicar la idea de Sadakane para evitar
		//los valores extremos mayores a 32 bits
		lenX = uint_len(bitX, 1);
		lenY = uint_len(bitY, 1);
		cdx = new uint[lenX]();
		cdy = new uint[lenY]();

		//Lleno cdx
		Gamma gamma;
		int posX = 0;
		for (i = 0; i < lenLog - 1; i++) {
			posX += gamma.encode(cdx, posX, x[i]);
		}

		//Lleno cdy

		int posY = 0;
		for (i = 0; i < lenLog - 1; i++) {
			posY += gamma.encode(cdy, posY, y[i]);
		}

		//como se elimina en el compact log el instante 0, se pone en 0 el bits
		//correspondiente en BsMov.
		bitclean(logRP->getBsMov(), 0);

		mov = BitSequenceFactory(logRP->getBsMov(), logRP->getLenMov());

		//assert((mov->countOnes())==(lenLog -1));

		delete[] x;
		x = NULL;
		delete[] y;
		y = NULL;
	} else {
		//se crea un CompactLog vacío
		cdx = NULL;
		cdy = NULL;
		mov = NULL;
		lenX = 0;
		lenY = 0;
	}
}

CompactLog::~CompactLog() {
	if (cdx != NULL) {
		delete[] this->cdx;
		cdx = NULL;
	}
	if (cdy != NULL) {
		delete[] this->cdy;
		cdy = NULL;
	}
	if (mov != NULL) {
		delete mov;
		mov = NULL;
	}

}

uint CompactLog::countSamples(uint i, uint j) {
	if (mov == NULL)
		return 0;
	if (i == 0) {
		if (j < (mov->getLength() - 1)) {
			return mov->rank1(j);
		} else {
			return mov->countOnes();
		}
	} else {
		if (j < (mov->getLength() - 1)) {
			return (mov->rank1(j) - mov->rank1(i - 1));
		} else {
			return (mov->countOnes() - mov->rank1(i - 1));
		}
	}
	return 0;
}

//-------------------------------------------------------------------------
//da verdadero si durante i,  j el punto está en el área de la consulta
int CompactLog::timeIntervalTest(uint x, uint y, uint i, uint j,
									AreaExpandida area) {
	Gamma gamma;
	ZigZagCoder coder;
	int resp = 0;
	if (mov == NULL) {
		//si el objeto no se ha movido, entonces
		return area.enAreaConsulta(x, y);
	}
	long int ri = mov->rank1(i);
	long int rj = mov->rank1(j);
	//si no hay movimientos hasta j, es equivalente a que mov==NULL
	if (rj == 0) {
		//si el objeto no se ha movido hasta j, entonces
		return area.enAreaConsulta(x, y);
	}
	//si ri=0, pero rj no. entonces hay un parte incial del intervalo de con
	// donde el punto no se ha movido, pero hay que revisar si su posición e
	if (ri == 0 && area.enAreaConsulta(x, y)) {
		return true;
	}

	//aquí rj, es el último punto que está dentro del intervalo, y por lo tanto
	//no se decodificarán mas alla del rj punto.
	long int p = 1;
	uint edx, edy, dx, dy, posX = 0, posY = 0;
	uint desp[4] = { 0, 0, 0, 0 };
	int particion;
	while (p <= rj) {
		//decodifico el punto actual
		posX += gamma.decode(cdx, posX, &edx);
		posY += gamma.decode(cdy, posY, &edy);
		dx = coder.decod(edx);
		dy = coder.decod(edy);
		//actualizo posición dado delta.
		x += dx;
		y += dy;
		//calculo los desplazamientos por ejes
		if (dx < 0) {
			desp[0] = -dx;
		} else {
			desp[1] = dx;
		}
		if (dy < 0) {
			desp[2] = -dy;
		} else {
			desp[3] = dy;
		}
		area.resize(desp);
		particion = area.getPartition(x, y);
		if (p < ri) {
			//significa que aún no se ha llegado al inicio del intervalo
			if (particion == UNIVERSO) {
				//en el caso que el punto se mueva a la partición universo
				//el punto no puede ser parte de los resultados
				return false;
			} else if (particion == INTERIOR) {
				//si está en la partición interior ya no saldrá.
				return true;
			}
		} else {
			//se encuentra dentro del intervalo
			if (particion == UNIVERSO) {
				//en el caso que el punto se mueva a la partición universo
				//el punto no puede ser parte de los resultados
				return false;
			} else if (particion == INTERIOR) {
				//si está en la partición interior ya no saldrá.
				return true;
			} else if (particion == MEDIA) {
				//significa que al menos  una vez entró en la región del
				//consulta y por lo tanto es parte de los resultados
				return true;
			}
		}
		p++;
	}
	//si llega a este punto significa que el objeto se quedó en la región
	//exterior, nunca entró al área de la consulta durante [i, j]
	return false;
}
//-------------------------------------------------------------------------
//
//
//da verdadero si en el instante i el punto está en el área de la consulta
int CompactLog::timeSliceTest(uint x, uint y, uint i, AreaExpandida area) {
	Gamma gamma;
	ZigZagCoder coder;
	int resp = 0;
	if (mov == NULL) {
		//si el objeto no se ha movido, entonces
		resp = area.getPartition(x, y);
		return ((resp == MEDIA) || (resp == INTERIOR));
	}
	int ri = mov->rank1(i);

	if (ri == 0) {
		//no se ha movido hasta i
		resp = area.getPartition(x, y);
		return ((resp == MEDIA) || (resp == INTERIOR));
	}

	//aquí ri, es el último punto que se debe decodificar
	int p = 1;
	uint edx, edy, dx, dy, posX = 0, posY = 0;
	uint desp[4] = { 0, 0, 0, 0 };
	int particion;
	while (p <= ri) {
		//decodifico el punto actual
		posX += gamma.decode(cdx, posX, &edx);
		posY += gamma.decode(cdy, posY, &edy);
		dx = coder.decod(edx);
		dy = coder.decod(edy);
		//actualizo posición dado delta.

		x += dx;
		y += dy;
		//printf("decod: Delta(%d, %d), Elias(%d, %d)\n", dx,dy,edx,edy);
		//calculo los desplazamientos por ejes
		if (dx < 0) {
			desp[0] = dx;
		} else {
			desp[1] = -dx;
		}
		if (dy < 0) {
			desp[2] = dy;
		} else {
			desp[3] = -dy;
		}

		area.resize(desp);
		particion = area.getPartition(x, y);
		if (p < ri) {
			//significa que aún no se ha llegado al instante
			if (particion == UNIVERSO) {
				//en el caso que el punto se mueva a la partición universo
				//el punto no puede ser parte de los resultados
				return false;
			} else if (particion == INTERIOR) {
				//si está en la partición interior ya no saldrá.
				return true;
			}
		} else {
			//se encuentra en el instante
			switch (particion) {
			case UNIVERSO:
			case EXTERIOR:
				return false;
			case MEDIA:
			case INTERIOR:
				return true;
			}
		}
		p++;
	}
	return false;
}
//----------------------------------------------------------------------------
//
//
Point * CompactLog::getTrajectory(uint oid, Point * pBase, uint tBase, uint i,
									uint j, LinkedList * resp) {
	int x = pBase->getX();
	int y = pBase->getY();

	if (mov == NULL) {
		//si el objeto no se ha movido, entonces
		return new Point(x, y);
	}

	long int ri = mov->rank1(i);
	long int rj = mov->rank1(j);
	//si no hay movimientos hasta j, es equivalente a que mov==NULL
	if (rj == 0) {
		//si el objeto no se ha movido hasta j, entonces
		return new Point(x, y);
	}
	if (ri == 0) {
		//si no hay puntos hasta el inicio del intervalo, pero como por lo menos
		//hay un punto en el intervalo, se avanza ri para que apunte al primer
		//1 dentro del intervalo
		ri++;
	}
	/*
	 * ri, representa al primera posición relativa dentro del intervalo, pero
	 * puede haber sido registrada un tiempo t anterior al intervalo, se
	 * recupera su instante con un select sobre mov. Recuperar el tiempo real
	 * donde ocurre es importante porque
	 *
	 */
	size_t t = mov->select1(ri);

	//por lo menos hay un punto que rescatar en el intervalo.
	Gamma gamma;
	ZigZagCoder zigzag;

	//aquí rj, es el último punto que está dentro del intervalo, y por lo tanto
	//no se decodificarán mas alla del rj punto.
	//count cuenta el punto que actualmente se está visitando partiendo del 1°
	uint count = 1;
	uint dx, dy, posX = 0, posY = 0;
	while (count <= rj) {
		//decodifico el punto actual
		posX += gamma.decode(cdx, posX, &dx);
		posY += gamma.decode(cdy, posY, &dy);
		//actualizo posición (x,y) dado delta.
		x += zigzag.decod(dx);
		y += zigzag.decod(dy);
		//se saltan los puntos desde 1° hasta ri-1 porque no están en el
		//intervalo, pero se deben decodificar igual.
		if (count >= ri) {
			//se encuentra dentro del intervalo
			add(resp,
				cElementKey(oid, cObjectTimePosition(oid, tBase + t, x, y)));
			t = mov->selectNext1(++t);
		}
		count++;
	}
	return new Point(x, y);
}
/**
 * Dado el instante j permite obtener el desplazamiento del
 * objeto como el par (x, y), donde x es el desplazamiento en el eje x e
 * y el desplazamiento en el eje y, desde el instante 1 hasta el instante j
 * pre: j dentro del rango 1..f
 **/
Point * CompactLog::getSumMov(uint j) {
	if (mov == NULL) {
		return new Point(0, 0);
	}

	size_t rj = mov->rank1(j);
	//por lo menos hay un punto que rescatar en el intervalo.
	Gamma gamma;
	ZigZagCoder zigzag;

	//aquí rj, es el último punto que está dentro del intervalo, y por lo tanto
	//no se decodificarán mas allá del rj punto.
	//count cuenta el punto que actualmente se está visitando partiendo del 1°
	uint count = 1;
	int x = 0, y = 0;
	uint dx = 0, dy = 0, posX = 0, posY = 0;
	while (count <= rj) {
		//decodifico el punto actual
		posX += gamma.decode(cdx, posX, &dx);
		posY += gamma.decode(cdx, posY, &dy);
		//actualizo posición dado delta.
		x += zigzag.decod(dx);
		y += zigzag.decod(dy);
		count++;
	}
	return new Point(x, y);
}

//--
//retorna un array con la suma parcial de los desplazamientos para cada instante
uint** CompactLog::getSumMov() {
	//el resultado es una matriz de len filas por 4 columnas
	if (mov == NULL) {
		return NULL;
	}
	int len = mov->getLength();
	uint ** resp = new uint*[len]();
	for (int i = 0; i < len; i++) {
		resp[i] = new uint[4]();
	}

	size_t rj = mov->countOnes();
	//por lo menos hay un punto que rescatar en el intervalo.
	Gamma gamma;
	ZigZagCoder coder;

	//aquí rj, es el último punto que está dentro del intervalo, y por lo tanto
	//no se decodificarán mas alla del rj punto.
	//count cuenta el punto que actualmente se está visitando partiendo del 1°
	uint count = 1;
	int x = 0, y = 0;
	uint dx = 0, dy = 0, posX = 0, posY = 0;
	//instante del primer 1.
	uint t = mov->select1(1);

	while (count <= rj) {
		//decodifico el punto actual
		posX += gamma.decode(cdx, posX, &dx);
		posY += gamma.decode(cdy, posY, &dy);
		//decodifico delta.
		x = coder.decod(dx);
		y = coder.decod(dy);
		//calculo los desplazamientos por ejes en valores absolutos.
		if (x < 0) {
			resp[t][0] = -x;
		} else {
			resp[t][1] = x;
		}
		if (y < 0) {
			resp[t][2] = -y;
		} else {
			resp[t][3] = y;
		}
		t = mov->selectNext1(++t);
		count++;
	}
	//calcula la suma parcial
	for (int i = 1; i < len; i++) {
		resp[i][0] += resp[i - 1][0];
		resp[i][1] += resp[i - 1][1];
		resp[i][2] += resp[i - 1][2];
		resp[i][3] += resp[i - 1][3];
	}
	return resp;

}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

size_t CompactLog::size() {

	size_t size = sizeof(CompactLog);
	if (mov == NULL) {
		return size;
	}
	size += this->mov->getSize();
	size += this->lenX;
	size += this->lenY;
	return size;
}

uint CompactLog::getLenTimeInterval() {
	return mov->getLength();
}

