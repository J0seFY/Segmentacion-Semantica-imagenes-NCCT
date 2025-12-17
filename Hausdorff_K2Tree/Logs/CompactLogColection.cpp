#include "CompactLogColection.h"

CompactLogColection::CompactLogColection() {
	this->totObj = 0;
	this->logArray = NULL;
	this->len = 0;

	maxPosDx = NULL;
	maxPosDy = NULL;

	maxNegDx = NULL;
	maxNegDy = NULL;

}

CompactLogColection::CompactLogColection(LogColection* lc) {
	this->totObj = lc->getTotObj();
	this->len = lc->getLenLog();
	if (totObj > 0) {
		this->logArray = new CompactLog*[totObj]();
		//visitar cada valor y crear la estructura compacta
		for (uint iter = 0; iter < totObj; iter++) {
			this->logArray[iter] = new CompactLog(lc->getLog(iter));
		}
		//hay que calcular el vector de máximo desplazamiento en x e y por instante.
		//

		maxPosDx = new uint[len]();
		maxPosDy = new uint[len]();
		maxNegDx = new uint[len]();
		maxNegDy = new uint[len]();

		uint ** desp = NULL;

		//esto es muy lento O(len*totObj), pero logra el mejor acotamiento.

		for (uint iter = 0; iter < totObj; iter++) {
			desp = logArray[iter]->getSumMov();
			if (desp != NULL) {
				for (uint i = 1; i < len; i++) {
					//len-1 = f
					if (desp[i][0] > maxNegDx[i]) {
						maxNegDx[i] = desp[i][0];
					}
					if (desp[i][2] > maxNegDy[i]) {
						maxNegDy[i] = desp[i][2];
					}
					if (desp[i][1] > maxPosDx[i]) {
						maxPosDx[i] = desp[i][1];
					}
					if (desp[i][3] > maxPosDy[i]) {
						maxPosDy[i] = desp[i][3];
					}
					delete[] desp[i];
				}
				delete[] desp;
			}
		}
	} else {
		logArray = NULL;
		maxPosDx = NULL;
		maxPosDy = NULL;
		maxNegDx = NULL;
		maxNegDy = NULL;
	}
}


CompactLogColection::~CompactLogColection() {
	if (logArray != NULL) {
		for (uint iter = 0; iter < totObj; iter++) {
			delete logArray[iter];
			logArray[iter] = NULL;
		}
		delete[] logArray;
		logArray = NULL;
	}

	if (maxPosDx != NULL) {
		delete[] maxPosDx;
		maxPosDx = NULL;
	}

	if (maxPosDy != NULL) {
		delete[] maxPosDy;
		maxPosDy = NULL;
	}

	if (maxNegDx != NULL) {
		delete[] maxNegDx;
		maxNegDx = NULL;
	}

	if (maxNegDy != NULL) {
		delete[] maxNegDy;
		maxNegDy = NULL;
	}
}

CompactLog * CompactLogColection::getCompatLog(uint o) {
	return (o < totObj) ? logArray[o] : NULL;
}

size_t CompactLogColection::size() {
	size_t size = sizeof(CompactLogColection);
	size += (8 * sizeof(uint) * len);	//los 8 array que almacenan los máximos desplazamientos.
	for (uint i = 0; i < totObj; i++) {
		size += logArray[i]->size();
	}
	return size;
}

//maximo desplazamiento desde i hasta len
//PRE: i está dentro del rango válido
uint * CompactLogColection::getMaxIzMovTo(uint i) {
	uint *resp = new uint[4]();
	if (i == 0) {
		return resp;
	}
	resp[0] = maxNegDx[i];
	resp[1] = maxPosDx[i];
	resp[2] = maxNegDy[i];
	resp[3] = maxPosDy[i];
	return resp;
}
