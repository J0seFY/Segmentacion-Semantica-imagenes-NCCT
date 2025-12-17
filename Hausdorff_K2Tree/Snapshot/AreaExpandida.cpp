/*
 * AreaExpandida.cpp
 *
 *  Created on: 17-07-2016
 *      Author: miguel
 */

#include "AreaExpandida.h"
//resta segura si (a-b) es < 0, la resta es 0.
#define restaS(a,b) b>a?0:a-b

AreaExpandida::AreaExpandida(uint x1, uint x2, uint y1, uint y2, uint MD[4]) {
	this->x1 = x1;
	this->x2 = x2;
	this->y1 = y1;
	this->y2 = y2;
	this->MD[0] = MD[1];
	this->MD[1] = MD[0];
	this->MD[2] = MD[3];
	this->MD[3] = MD[2];
}
void AreaExpandida::resize(uint des[4]) {
	//el máximo desplazamiento por eje cambia, porque
	//ya se han realizado algunos de los que se esperaban
	//ahora quedan menos.
	this->MD[0] -= des[1];
	this->MD[1] -= des[0];
	this->MD[2] -= des[3];
	this->MD[3] -= des[2];
}
AreaExpandida::~AreaExpandida() {
	// TODO Auto-generated destructor stub
}

//dado un rectángulo paralelo a los ejes entrega verdadero si el punto está
//dentro del rectángulo, sino da falso
inline int AreaExpandida::enRectangulo(long int x, long int y, long int x1,
										long int x2, long int y1, long int y2) {
	return (x >= x1 && x <= x2 && y >= y1 && y <= y2);

}
//Retorna verdadero si el punto x,y está dentro del área original de la consulta
int AreaExpandida::enAreaConsulta(uint x, uint y) {
	return enRectangulo(x, y, x1, x2, y1, y2);
}
//Entrega la partición en la que se encuentra el punto en el área de la consulta
int AreaExpandida::getPartition(uint x, uint y) {

	/*
	 * Si D es 0, entonces, no existe partición exterior, y
	 * todos los elementos que están fuera del rango son parte
	 * del universo, es decir, nunca pueden entrar al rango
	 * de la consulta.
	 *
	 * si D es mayor a la mitad del ancho de la ventana espacial, entonces no
	 * hay región interior
	 *
	 * el otro caso especial es que la región exterior o la región universo
	 * no exista porque D sea muy grande o bien la ventana espacial es del
	 * ancho del espacio total
	 */

	long int dx1 = MD[0], dx2 = MD[1], dy1 = MD[2], dy2 = MD[3];

	if (!(enRectangulo(x, y, restaS(x1, dx1), x2 + dx2, restaS(y1, dy1),
						y2 + dy2))) {
		return UNIVERSO;
	} else if (!(enRectangulo(x, y, x1, x2, y1, y2))) {
		return EXTERIOR;
	} else if (((x1 + dx2) >= ((restaS(x2, dx1))))
			|| ((y1 + dy2) >= ((restaS(y2, dy1))))) {
		//si no existe la partición interior
		return MEDIA;
	} else if (!(enRectangulo(x, y, x1 + dx2, restaS(x2, dx1), y1 + dy2,
								restaS(y2, dy1)))) {
		return MEDIA;
	} else {
		return INTERIOR;
	}

	return UNIVERSO;
}

