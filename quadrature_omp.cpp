//---------------------------------------------------------------------
// In this code, the value of pi is computed in two ways:
//
// 1) Using the trapezoidal rule for the numerical integration of 4/(1+x*x)
// from 0 to 1 using variable number of steps.
// 
// 2) Using a MonteCarlo simulation with random numbers to evalute the area
// of a unit circle.
//
// Both methods are parallelized with OpenMP.
//
// Written by William Beordo.
//---------------------------------------------------------------------

#include <iomanip>
#include <iostream>
#include <cmath>
#include <ctime>
#include <fstream>
#include "StopWatch.h"
#include <omp.h>

double TrapezoidalQuad(double(*Func)(double), double, double, int);
double Function(double);
double MonteCarlo(int);

using namespace std;

int main()
{   
	cout << setiosflags(ios::scientific);
	
	double a = 0.0; // left boundary of integration
	double b = 1.0; // right boundary of integration
	long int n = 5000000000; // number of integration intervals for the Trapezoidal rule 
                             // and number of points for the MonteCarlo sampling.
    int max_nthreads = 16; // maximum number of threads
    
    for(int i=1; i<=max_nthreads; i++)
    {
        StopWatch stopWatch;
        
        omp_set_num_threads(i); // setting the number of threads for the parallelized computation
        double integral = TrapezoidalQuad(Function, a, b, n);
        double mc = MonteCarlo(n);
        cout << "Pi with trapezoidal quadrature: " << setprecision(12) << integral 
             << " \t Pi with montecarlo: " << setprecision(6) << mc 
             << " \t for " << i << " THREAD(s)" << endl;
    }
	
	return 0;
}

// Trapezoidal rule of numerical quadrature. The integrand is passed by Function()
double TrapezoidalQuad(double(*Func)(double x), double xa, double xb, int n)
{
	double dx = (xb-xa)/double(n); // integration step
	double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum) // splitting the integration among threads, avoiding race conditions 
                                              // for the variable 'sum'
	for(long int i=0; i<n; i++)
	{
		double x = xa + i*dx;
		sum += 0.5*(Func(x) + Func(x+dx));
	}
    
	return sum*dx;
}

// Function to integrate
double Function(double x)
{
	return 4.0/(1 + x*x); // integrand for the Pi computation 
}

// 2-dim integration of a unit circle, using random sampling
double MonteCarlo(int n)
{
    double pi;
    int count=0;

    #pragma omp parallel // forking the following scope among threads
    {                  
        unsigned int myseed = omp_get_thread_num(); // assigning a seed to each thread for random number generation
        #pragma omp for reduction (+: count) // splitting the sampling among threads avoiding race conditions
        for(int i = 0; i < n; i++)
        {
            if(pow((double) rand_r(&myseed) / (RAND_MAX), 2) + pow((double) rand_r(&myseed) / (RAND_MAX), 2) <= 1)
               count++; // only random points inside the circle are counted
        }
    }

    pi = 4*(double)count/n; // 4 is the area of the squared enclosing the circle
	
	return pi;
}