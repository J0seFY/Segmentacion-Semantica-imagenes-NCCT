/********************************************************************************
 Modificado del código de Francisco Claude en libcds2.
 se ha corregido error.  Se ha migrado a libcds1.

 Copyright (c) 2012, Francisco Claude.
 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 * Neither the name of libcds nor the names of its
 contributors may be used to endorse or promote products derived from this
 software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************/

#include "PermMRRR.h"

#include <algorithm>
#include <vector>

using std::vector;

PermMRRR::PermMRRR(uint *perm, uint len, uint rate) {
	permutation_ = perm;
	len_ = len;
	//bitmap para marcar cuando se visita un elemento de la permutación
	uint *visited = new uint[uint_len(len, 1)]();

	//bitmap para marcar cuando un elemento de la permutación es sampleado
	uint *sampled = new uint[uint_len(len, 1)]();

	vector<uint> sampled_values;
	vector<uint> sampled_values_pos;
	for (uint i = 0; i < len; i++) {
		if (bitget(visited,i)) {
			continue;
		}
		bool added = false;
		uint last = i;
		bitset(visited, i);
		uint pos = perm[i];
		uint count = 1;
		while (pos != i) {
			bitset(visited, pos);
			if (count == rate) {
				added = true;
				sampled_values.push_back(last);
				sampled_values_pos.push_back(pos);
				bitset(sampled, pos);
				last = pos;
				count = 0;
			}
			pos = perm[pos];
			count++;
		}
		if (added) {
			bitset(sampled,pos);
			sampled_values.push_back(last);
			sampled_values_pos.push_back(pos);
		}
	}

	//ordenando adecuadamente samples_values
	sampled_ = BitSequenceFactory(sampled,len);

	uint lenArreglo=sampled_->countOnes();
	uint * arreglo=new uint[lenArreglo]();
	uint pos;
	for (uint i = 0; i < lenArreglo; i++) {
		pos = sampled_->rank1(sampled_values_pos[i])-1;
		arreglo[pos] = sampled_values[i];
	}

	rev_links_ = arreglo;

	delete[] visited;
	delete[] sampled;
}

PermMRRR::~PermMRRR() {
	delete[] rev_links_;
	rev_links_=NULL;
	delete[] permutation_;
	permutation_=NULL;
	delete sampled_;
	sampled_=NULL;
}

uint PermMRRR::Access(uint i) const {
	return permutation_[i];
}

uint PermMRRR::Reverse(uint i) const {
	uint pos = i;
	uint next1 = permutation_[pos];
	while (next1 != i) {
		if (sampled_->access(pos)) {
			pos = rev_links_[(sampled_->rank1(pos) - 1)];
			uint next = permutation_[pos];
			while (next != i) {
				pos = next;
				next = permutation_[next];
			}
			return pos;
		}
		pos = next1;
		next1 = permutation_[pos];
	}
	return pos;
}

uint PermMRRR::GetSize() const {
	return len_*sizeof(uint) + sampled_->getSize() + sampled_->countOnes()*sizeof(uint);
}

uint PermMRRR::GetLength() const {
	return len_;
}

