/*
 * Point.h
 *
 *  Created on: 09-10-2013
 *      Author: miguel
 */

#ifndef POINT_H_
#define POINT_H_
#include <string.h>
#include <iostream>

class Point {
public:
	Point();
	Point(int x, int y);
	 int getX();
	 int getY();
	void setX(int x);
	void setY(int y);
	void setXY(int x, int y);
	Point * sumar(Point * p);
	Point * restar(Point * p);
    void print() const {
        std::cout << "(" << x << ", " << y << ")" << std::endl;
    }
	virtual ~Point();

public:
	 int x;
	 int y;
};

#endif /* POINT_H_ */
