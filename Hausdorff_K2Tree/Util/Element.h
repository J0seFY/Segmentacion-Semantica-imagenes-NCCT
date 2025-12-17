#ifndef ELEMENT_INCLUDED
#define ELEMENT_INCLUDED

//      Element.h
//      
//      Copyright 2011 Miguel Romero <miguel@miguel-laptop>
//      
//      This program is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//      
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//      
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//      MA 02110-1301, USA.

typedef struct sElement {
	int key;
	void * pt;   //puntero genérico al dato almacenado en Element
} Element;

//constructor de la estructura
Element * cElementEmpty();

Element * cElement(void * pt);
Element * cElementKey(int key, void * pt);

//destructor de la estructura
void destroy(Element * e);

//función observadora
void * get(Element * e);
int getKey(Element * e);
//función mutadora
void set(Element * e, void * pt);
void setKey(Element * e, int key);

long int * estadisticaElement();
#endif
