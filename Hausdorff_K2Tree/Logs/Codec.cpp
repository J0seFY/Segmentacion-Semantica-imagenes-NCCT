/*
 * Codec.cpp
 * zigzag coding.
 * bitwise version based on:
 * http://lemire.me/blog/2012/02/08/effective-compression-using-frame-of-reference-and-delta-coding/
 *  Created on: 17-10-2013
 *      Author: miguel
 */

#include "Codec.h"

ZigZagCoder::ZigZagCoder() {
	// TODO Auto-generated constructor stub

}

ZigZagCoder::~ZigZagCoder() {
	// TODO Auto-generated destructor stub
}
int ZigZagCoder::cod(int x) {
	//se usa un diccionario fijo, donde el ranking por frecuencia es:
	//0, -1, 1, -2, 2, -3, 3... N.
	//así el signo indica si el número es par o impar y el valor es el término
	//de la serie de pares o impares.
	//el código es la posicion en la secuencia de números naturales.
	/* este código es lo mismo que hace el return, pero menos eficiente
	if (value == 0) {
		return 0;
	} else if (value > 0) {
		//es par
		return value * 2;
	}
	//es impar
	return (value * -2) - 1;
	*/
	return (x << 1) ^ (x >> 31);
}
int ZigZagCoder::decod(int y) {
	//decod sigzag.
	/*este código es lo que hace el return, pero menos eficiente
	if (value == 0) {
		return 0;
	}else if (value%2==0){
		//es par
		return value/2;
	}
	//es impar
	return -((value/2)+1);
	*/

	return (((uint)y >> 1) ^ ((y << 31) >> 31));
}
