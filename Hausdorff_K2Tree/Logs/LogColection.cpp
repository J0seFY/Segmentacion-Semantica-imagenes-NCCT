#include "LogColection.h"

LogColection::LogColection(int totObj, int lenLog) {
	this->logArray = new LogRP*[totObj]();
	this->totObj = totObj;
	this->lenLog = lenLog;
	for (int iter = 0; iter < totObj; iter++) {
		logArray[iter] = new LogRP(lenLog);
	}
}
LogColection::~LogColection() {
	//ver bien que pasa con esta eliminación y el objeto compartido
	//mov de los LogRP
	for (int iter = 0; iter < totObj; iter++) {
		delete logArray[iter];
		logArray[iter]=NULL;
	}
	delete[] logArray;
	logArray=NULL;
}
//------------------------------------------------------------------------
//tamaño del log.
size_t LogColection::size() {
	size_t resp = sizeof(LogColection);
	for (int iter = 0; iter < totObj; iter++) {
		resp+=logArray[iter]->size();
	}
	return resp;
}
//----------------------------------------------------------------------------
//almacenar la posición relativa (dx,dy) para el objeto o, en la posición i
//
//
void LogColection::setChange(uint o, int dx, int dy, uint i) {
	//se ignoran los objetos con ID por sobre el total de objetos
	if (o < totObj) {
		logArray[o]->setChange(dx, dy, i);
	}
}

LogRP * LogColection::getLog(uint o) {
	return logArray[o];
}
