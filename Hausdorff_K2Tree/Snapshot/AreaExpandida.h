/*
 * AreaExpandida.h
 *
 *  Created on: 17-07-2016
 *      Author: miguel
 */

#ifndef SNAPSHOT_AREAEXPANDIDA_H_
#define SNAPSHOT_AREAEXPANDIDA_H_
#define UNIVERSO 1
#define EXTERIOR 2
#define MEDIA 3
#define INTERIOR 4

#ifndef uint
#define uint unsigned int
#endif

class AreaExpandida {
public:
	AreaExpandida(uint x1, uint x2, uint y1, uint y2,uint MD[4]);
	virtual ~AreaExpandida();
	int getPartition(uint x, uint y);
	void resize(uint des[4]);
	int enAreaConsulta(uint x, uint y);
private:
	//rango espacial
	uint x1;
	uint x2;
	uint y1;
	uint y2;
	uint MD[4]; //cuatro valores
	inline int enRectangulo(long int x, long int y, long int x1, long int x2,
								long int y1, long int y2);
};

#endif /* SNAPSHOT_AREAEXPANDIDA_H_ */
