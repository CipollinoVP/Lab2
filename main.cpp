#include "iostream"
#include <cmath>
#include <omp.h>

using namespace std;

const double PI = 3.141592653589793238;
const double EPS = 1e-6;

//Правая часть
double f(double x, double y, double k)
{
    return 2.0 * sin(PI * y) + k * k * (1.0 - x) * x * sin(PI * y) + PI * PI * (1.0 - x) * x * sin(PI * y);
}

//Точное решение
double u_analitic(double x, double y)
{
    return x * (1 - x) * sin(PI * y);
}

//Норма разности векторов
double norma_vect(int N, double* v1, double* v2, int th)
{
    double temp=0.0;
    double norm = 0.0;
#pragma omp parallel for num_threads(th) if (th > 1)
    for (int i = 0; i < N; ++i)
    {
        temp = v1[i] - v2[i];
        //#pragma omp critical
        if (fabs(temp) > norm)
            norm = fabs(temp);
    }
    return norm;
}

//Метод Якоби
void method_Jacobi(int N, double* y, double* yp, double* f1, double h, double k, double c1, double c2, int th)
{
#pragma omp parallel for num_threads(th) if (th > 1)
    for (int i = 0; i < N * N; ++i)
        y[i] = 0.0;

    //#pragma omp parallel for num_threads(th) if (th > 1)
    //for (int i = 0; i < N; ++i)
    //{
    //for (int j = 0; j < N; ++j)
    //f1[i * N + j] = c2 * f(i * h, j * h, k);
    //}

    int iter = 0;
    double norma = 0.0;

    do {
        ++iter;

        //#pragma omp parallel for num_threads(th) if (th > 1)
        //for (int i = 0; i < N * N; ++i)
        //yp[i] = y[i];
        swap(y, yp);

#pragma omp parallel for num_threads(th) if (th > 1)
        for (int i = 1; i < N - 1; ++i)
            for (int j = 1; j < N - 1; ++j)
                y[i * N + j] = c1 * (yp[(i + 1) * N + j] + yp[(i - 1) * N + j] + yp[i * N + j + 1] + yp[i * N + j - 1]) + c2 * f(i * h, j * h, k);

        norma = norma_vect(N * N, y, yp, th);

    } while (norma > EPS);

    cout << "Количество потоков = " << th << endl;
    cout<< "Количество итераций = " << iter << endl;
    cout << "Норма yp - y  = " << norma << endl;
}

//Метод красно-чёрных итераций
void red_black_iterations(int N, double* y, double* yp, double* f1, double h, double k, double c1, double c2, int th)
{
#pragma omp parallel for num_threads(th) if (th > 1)
    for (int i = 0; i < N * N; ++i)
        y[i] = 0.0;

    //#pragma omp parallel for num_threads(th) if (th > 1)
    //for (int i = 0; i < N; ++i)
    //{
    //for (int j = 0; j < N; ++j)
    //f1[i * N + j] = c2 * f(i * h, j * h, k);
    //}

    int iter = 0;
    double norma = 0.0;

    do {
        ++iter;

        //#pragma omp parallel for num_threads(th) if (th > 1)
        //for (int i = 0; i < N * N; ++i)
        //yp[i] = y[i];
        swap(y, yp);

#pragma omp parallel for num_threads(th) if (th > 1)
        for (int i = 1; i < N - 1; ++i)
            for (int j = i % 2 + 1; j < N - 1; j = j + 2)
                y[i * N + j] = c1 * (yp[(i + 1) * N + j] + yp[(i - 1) * N + j] + yp[i * N + j + 1] + yp[i * N + j - 1]) + c2 * f(i * h, j * h, k);

#pragma omp parallel for num_threads(th) if (th > 1)
        for (int i = 1; i < N - 1; ++i)
            for (int j = (i + 1) % 2 + 1; j < N - 1; j = j + 2)
                y[i * N + j] = c1 * (y[(i + 1) * N + j] + y[(i - 1) * N + j] + y[i * N + j + 1] + y[i * N + j - 1]) + c2 * c2 * f(i * h, j * h, k);

        norma = norma_vect(N * N, y, yp, th);

    } while (norma > EPS);

    cout << "Количество потоков = " << th << endl;
    cout << "Количество итераций = " << iter << endl;
    cout << "Норма yp - y  = " << norma << endl;
}

int main()
{
    setlocale(LC_ALL, "Russian");

    double x_l = 0.0;
    double x_r = 1.0;
    double y_l = 0.0;
    double y_r = 1.0;

    int N_frag = 4000; //Количество разбиений
    int N = N_frag + 1; //Количество узлов
    double h = (x_r - x_l) / N_frag; //Шаг
    double k;
    k = 3.0 / h;
    double c1, c2;
    c1 = 1.0 / (4.0 + h * h * k * k);
    c2 = h * h * c1;
    double th; //Количество потоков

    auto* y = new double[N * N]; //Численное решение в узле i, j
    auto* yp = new double[N * N]; //Решение на следующем слое
    auto* u = new double[N * N]; //Точное решение
    auto* f1 = new double[N * N];//Вектор правой части

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            u[i * N + j] = u_analitic(i * h, j * h);

    cout << "N = " << N_frag << endl << endl;
    cout << "Метод Якоби" << endl;
    double t1, t2, time1;
    t1 = omp_get_wtime();
    method_Jacobi(N, y, yp, f1, h, k, c1, c2, 1);
    t2 = omp_get_wtime();
    time1 = (double)(t2 - t1);
    cout << "Норма u - y = " << norma_vect(N * N, y, u, th) << endl;
    cout << "Время выполнения  = " << time1 << " с" << endl << endl;

    cout << "Метод красно-черных итераций" << endl;
    double t3, t4, time2;
    t3 = omp_get_wtime();
    red_black_iterations(N, y, yp, f1, h, k, c1, c2, 1);
    t4 = omp_get_wtime();
    time2 = (double)(t4 - t3);
    cout << "Норма u - y = " << norma_vect(N * N, y, u, th) << endl;
    cout << "Время выполнения  = " << time2 << " с" << endl << endl;

    cout << "Метод Якоби параллельный" << endl;
    double t5, t6, time3;
    t5 = omp_get_wtime();
    method_Jacobi(N, y, yp, f1, h, k, c1, c2, 18);
    t6 = omp_get_wtime();
    time3 = (double)(t6 - t5);
    cout << "Норма u - y = " << norma_vect(N * N, y, u, th) << endl;
    cout << "Время выполнения  = " << time3 << " с" << endl << endl;

    cout << "Метод красно-черных итераций параллельный" << endl;
    double t7, t8, time4;
    t7 = omp_get_wtime();
    red_black_iterations(N, y, yp, f1, h, k, c1, c2, 18);
    t8 = omp_get_wtime();
    time4 = (double)(t8 - t7);
    cout << "Норма u - y = " << norma_vect(N * N, y, u, th) << endl;
    cout << "Время выполнения  = " << time4 << " с" << endl << endl;

    cout << "Ускорение Метод Якоби/Метод Якоби параллельный = " << (double)time1 / (double)time3 << endl;
    cout << "Ускорение Метод красно - черных итераций/Метод красно-черных итераций параллельный = " << (double)time2 / (double)time4 << endl;

    delete[] y;
    delete[] yp;
    delete[] u;
    delete[] f1;
    return 0;
}

