//
// Created by fernando on 05-07-24.
//

#include <vector>
#include <limits>
#include <algorithm>
#include "utils.h"
#include <cmath>
#include <Util/Point.h>
#include "ryu-kamata.h"

using namespace std;

double distanceKm(Point p1, Point p2) {
    return sqrt(pow(p1.getX() - p2.getX(), 2) + pow(p1.getY() - p2.getY(), 2));
}

void tmdc(const std::vector<Point> &A, const std::vector<Point> &B, vector<double> &U, vector<double> &V, double &cmax, int lambda) {
    int delta = B.size() / static_cast<double>(lambda); // Adjust delta based on lambda (λ) parameter
    int osp = generarNumeroAleatorio(0, delta); // Generate random starting point within δ - 1 range
    int isp = A.size() / 2;
    double dist = 0.0;
    for (int x = osp; x < B.size(); x += delta) {
        double cmin = numeric_limits<double>::infinity();
        double dist1 = numeric_limits<double>::infinity();
        double dist2 = numeric_limits<double>::infinity();

        for (int y = isp - 1, z = isp; y > 0 || z < A.size(); y--, z++) {
            if (y >= 0) {
                dist1 = distanceKm(B[x], A[y]);
                if (dist1 < U[y]) {
                    U[y] = dist1;
                }
            }

            if (z < A.size()) {
                dist2 = distanceKm(B[x], A[z]);
                if (dist2 < U[z]) {
                    U[z] = dist2;
                }
            }

            dist = min(dist1, dist2);
            if (dist < V[x]) {
                V[x] = dist;
            }
            cmin = min(dist, cmin);
            if (cmin <= cmax) {
                isp = dist1 < dist2 ? y : z;
                break;
            }
        }

        if (cmin > cmax) {
            cmax = cmin;
        }
    }
}

std::vector<Point> rulingOut(const std::vector<Point> &S, const std::vector<double> &W, double cmax) {
    std::vector<Point> Sro;
    for (size_t i = 0; i < S.size(); i++) {
        if (W[i] > cmax) {
            Sro.push_back(S[i]);
        }
    }
    return Sro;
}

pair<double, vector<double>> directedHD(const vector<Point> &A, const vector<Point> &B, vector<double> &U, vector<double> &V) {
    double cmax = 0;
    int isp = B.size() / 2;

    vector<Point> Aro = rulingOut(A, U, cmax); // Perform rulingout step

    for (int x = 0; x < Aro.size(); x++) {
        double cmin = numeric_limits<double>::infinity();
        double dist1 = numeric_limits<double>::infinity();
        double dist2 = numeric_limits<double>::infinity();

        for (int y = isp - 1, z = isp; y > 0 || z < B.size(); y--, z++) {
            if (y >= 0) {
                dist1 = distanceKm(Aro[x], B[y]);
                if (dist1 < V[y]) {
                    V[y] = dist1;
                }
            }
            if (z < B.size()) {
                dist2 = distanceKm(Aro[x], B[z]);
                if (dist2 < V[z]) {
                    V[z] = dist2;
                }
            }
            cmin = min(min(dist1, dist2), cmin);
            if (cmin <= cmax) {
                isp = dist1 < dist2 ? y : z;
                break;
            }
        }

        if (cmin > cmax) {
            cmax = cmin;
        }
    }
    return {cmax, V}; // Return cmax and updated V
}

double directHausdorff(const vector<Point> &A, const vector<Point> &B, int lambda){
    double cmax = 0;
    std::vector<double> U(A.size(), numeric_limits<double>::infinity()); //lista de minimos temporales de A
    std::vector<double> V(B.size(), numeric_limits<double>::infinity()); //lista de minimos temporales de B
    tmdc(A, B, U, V, cmax, lambda);

    auto A_B = directedHD(A, B, U, V);
    return A_B.first;
}

double hausKamata(const vector<Point> &A, const vector<Point> &B, int lambda) {

    double cmax = 0;
    std::vector<double> U(A.size(), numeric_limits<double>::infinity()); //lista de minimos temporales de A
    std::vector<double> V(B.size(), numeric_limits<double>::infinity()); //lista de minimos temporales de B
    tmdc(A, B, U, V, cmax, lambda);

    // BUGFIX (2025-11-18): Se corrige problema de double free/corruption
    // Problema original: directedHD retorna pair<double, vector<double>> por VALOR (crea copia),
    // y al reutilizar A_B.second en la segunda llamada se generaba aliasing de memoria
    // porque el vector tenia tamaño B.size() pero se esperaba tamaño A.size().
    // Solucion: Crear vectores independientes U2 y V2 para la segunda direccion.
    
    auto A_B = directedHD(A, B, U, V);
    
    // Crear nuevos vectores temporales para calcular distancia en direccion opuesta (B -> A)
    // U2 debe tener tamaño B (conjunto origen) y V2 tamaño A (conjunto destino)
    std::vector<double> U2(B.size(), numeric_limits<double>::infinity());
    std::vector<double> V2(A.size(), numeric_limits<double>::infinity());
    
    auto B_A = directedHD(B, A, U2, V2);

   // cout << "direct hauss de a - b: " << A_B.first << endl;
   // cout << "direct hauss de b - a: " << B_A.first << endl;

    return  max(A_B.first, B_A.first);
}