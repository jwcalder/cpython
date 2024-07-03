/* synchronization.c - C code acceleration for synchronization
 *
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include "vector_operations.h"
#include "memory_allocation.h"


double** setup_2D_array(double *data, int m, int n);
double*** setup_3D_array(double *data, int m, int n, int p);
void sum(double ***u, double ***v, double ***w, int m, int n, int p);
void multiply(double ***u, double ***v, double ***w, int m, int n, int p);
void transpose_matmult(double ***u, double ***v, double ***w, int m, int n, int p);
void subtract(double ***u, double ***v, double ***w, int m, int n, int p);
void t_matmult(double **u, double **v, double **w, int n, int p);

//Sets up a block fo data so it can be accessed as a 2D array
double** setup_2D_array(double *data, int m, int n){
   double **ptr = (double**)malloc(m*sizeof(double*));
   ptr[0] = data;
   int i;
   for(i=0;i<m;i++){
      ptr[i] = ptr[0] + n*i;
   }
   return ptr;
}
//Sets up a block fo data so it can be accessed as a 3D array
double*** setup_3D_array(double *data, int m, int n, int p){
   double ***ptr = (double***)malloc(m*sizeof(double**));
   int i;
   for(i=0;i<m;i++){
      ptr[i] = setup_2D_array(data+i*n*p,n,p);
   }
   return ptr;
}
void sum(double ***u, double ***v, double ***w, int m, int n, int p){
   int i,j,k;
   for(i=0;i<m;i++){
      for(j=0;j<n;j++){
         for(k=0;k<p;k++){
            w[i][j][k] = u[i][j][k] + v[i][j][k];
         }
      }
   }
}
void subtract(double ***u, double ***v, double ***w, int m, int n, int p){
   int i,j,k;
   for(i=0;i<m;i++){
      for(j=0;j<n;j++){
         for(k=0;k<p;k++){
            w[i][j][k] = u[i][j][k] - v[i][j][k];
         }

      }
   }
}
void multiply(double ***u, double ***v, double ***w, int m, int n, int p){
   int i,j,k;
   for(i=0;i<m;i++){
      for(j=0;j<n;j++){
         for(k=0;k<p;k++){
            w[i][j][k] = u[i][j][k] * v[i][j][k];
         }
      }
   }
}
void divide(double ***u, double ***v, double ***w, int m, int n, int p){
   int i,j,k;
   for(i=0;i<m;i++){
      for(j=0;j<n;j++){
         for(k=0;k<p;k++){
            w[i][j][k] = u[i][j][k] / v[i][j][k];
         }
      }
   }
}
void transpose_matmult(double ***u, double ***v, double ***w, int m, int n, int p){
   int i;
   for(i=0;i<m;i++){
      t_matmult(u[i],v[i],w[i],n,p);
   }
}
//Matrix multiplication of u.T@v, stored in w
void t_matmult(double **u, double **v, double **w, int n, int p){
   int i,j,k;
   for(i=0;i<p;i++){
      for(j=0;j<p;j++){
         w[i][j] = 0;
         for(k=0;k<n;k++){
            w[i][j] += u[k][i]*v[k][j];
         }
      }
   }
}

//This routine parses the input data from Python and calls the sum function
static PyObject* add_arrays(PyObject* self, PyObject* args)
{

   PyArrayObject *u_array;
   PyArrayObject *v_array;
   PyArrayObject *w_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &u_array, &PyArray_Type, &v_array, &PyArray_Type, &w_array))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(u_array);
   int m = dim[0]; //Number of rows in array
   int n = dim[1]; //Number of columns in array
   int p = dim[2]; //Number of ... in array

   //Pointers to block of data that hold the arrays
   double *u_data = (double *) PyArray_DATA(u_array);
   double *v_data = (double *) PyArray_DATA(v_array);
   double *w_data = (double *) PyArray_DATA(w_array);
   
   //Set up as 3D arrays in C
   double ***u = setup_3D_array(u_data,m,n,p);
   double ***v = setup_3D_array(v_data,m,n,p);
   double ***w = setup_3D_array(w_data,m,n,p);

   sum(u,v,w,m,n,p);

   Py_INCREF(Py_None);
   return Py_None;
}
//This routine parses the input data from Python and calls the subtract function
static PyObject* subtract_arrays(PyObject* self, PyObject* args)
{

   PyArrayObject *u_array;
   PyArrayObject *v_array;
   PyArrayObject *w_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &u_array, &PyArray_Type, &v_array, &PyArray_Type, &w_array))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(u_array);
   int m = dim[0]; //Number of rows in array
   int n = dim[1]; //Number of columns in array
   int p = dim[2]; //Number of ... in array

   //Pointers to block of data that hold the arrays
   double *u_data = (double *) PyArray_DATA(u_array);
   double *v_data = (double *) PyArray_DATA(v_array);
   double *w_data = (double *) PyArray_DATA(w_array);
   
   //Set up as 3D arrays in C
   double ***u = setup_3D_array(u_data,m,n,p);
   double ***v = setup_3D_array(v_data,m,n,p);
   double ***w = setup_3D_array(w_data,m,n,p);

   subtract(u,v,w,m,n,p);

   Py_INCREF(Py_None);
   return Py_None;
}
//This routine parses the input data from Python and calls the mult function
static PyObject* multiply_arrays(PyObject* self, PyObject* args)
{

   PyArrayObject *u_array;
   PyArrayObject *v_array;
   PyArrayObject *w_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &u_array, &PyArray_Type, &v_array, &PyArray_Type, &w_array))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(u_array);
   int m = dim[0]; //Number of rows in array
   int n = dim[1]; //Number of columns in array
   int p = dim[2]; //Number of ... in array

   //Pointers to block of data that hold the arrays
   double *u_data = (double *) PyArray_DATA(u_array);
   double *v_data = (double *) PyArray_DATA(v_array);
   double *w_data = (double *) PyArray_DATA(w_array);
   
   //Set up as 3D arrays in C
   double ***u = setup_3D_array(u_data,m,n,p);
   double ***v = setup_3D_array(v_data,m,n,p);
   double ***w = setup_3D_array(w_data,m,n,p);

   multiply(u,v,w,m,n,p);

   Py_INCREF(Py_None);
   return Py_None;
}
//This routine parses the input data from Python and calls the divide function
static PyObject* divide_arrays(PyObject* self, PyObject* args)
{

   PyArrayObject *u_array;
   PyArrayObject *v_array;
   PyArrayObject *w_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &u_array, &PyArray_Type, &v_array, &PyArray_Type, &w_array))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(u_array);
   int m = dim[0]; //Number of rows in array
   int n = dim[1]; //Number of columns in array
   int p = dim[2]; //Number of ... in array

   //Pointers to block of data that hold the arrays
   double *u_data = (double *) PyArray_DATA(u_array);
   double *v_data = (double *) PyArray_DATA(v_array);
   double *w_data = (double *) PyArray_DATA(w_array);
   
   //Set up as 3D arrays in C
   double ***u = setup_3D_array(u_data,m,n,p);
   double ***v = setup_3D_array(v_data,m,n,p);
   double ***w = setup_3D_array(w_data,m,n,p);

   divide(u,v,w,m,n,p);

   Py_INCREF(Py_None);
   return Py_None;
}
//
//This routine parses the input data from Python and calls the divide function
static PyObject* transpose_matmult_arrays(PyObject* self, PyObject* args)
{

   PyArrayObject *u_array;
   PyArrayObject *v_array;
   PyArrayObject *w_array;

   /*  parse arguments */
   if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &u_array, &PyArray_Type, &v_array, &PyArray_Type, &w_array))
      return NULL;

   npy_intp *dim =  PyArray_DIMS(u_array);
   int m = dim[0]; //Number of rows in array
   int n = dim[1]; //Number of columns in array
   int p = dim[2]; //Number of ... in array

   //Pointers to block of data that hold the arrays
   double *u_data = (double *) PyArray_DATA(u_array);
   double *v_data = (double *) PyArray_DATA(v_array);
   double *w_data = (double *) PyArray_DATA(w_array);
   
   //Set up as 3D arrays in C
   double ***u = setup_3D_array(u_data,m,n,p);
   double ***v = setup_3D_array(v_data,m,n,p);
   double ***w = setup_3D_array(w_data,m,p,p);

   transpose_matmult(u,v,w,m,n,p);

   Py_INCREF(Py_None);
   return Py_None;
}
//Everything below is the python wrapper

/*  define functions in module */
static PyMethodDef SynchronizationMethods[] =
{
   {"add_arrays", add_arrays, METH_VARARGS, "C Code acceleration for adding arrays"},
   {"multiply_arrays", multiply_arrays, METH_VARARGS, "C Code acceleration for multiplying arrays"},
   {"subtract_arrays", subtract_arrays, METH_VARARGS, "C Code acceleration for subtracting arrays"},
   {"divide_arrays", divide_arrays, METH_VARARGS, "C Code acceleration for dividing arrays"},
   {"transpose_matmult_arrays", transpose_matmult_arrays, METH_VARARGS, "C Code acceleration for matrix multiplication"},
   {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem =
{
   PyModuleDef_HEAD_INIT,
   "synchronization", "Some documentation",
   -1,
   SynchronizationMethods
};

PyMODINIT_FUNC
PyInit_synchronization(void)
{
   import_array();
   return PyModule_Create(&cModPyDem);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC
initsynchronization(void)
{
   (void) Py_InitModule("synchronization", SynchronizationMethods);
   /* IMPORTANT: this must be called */
   import_array();
}
#endif




