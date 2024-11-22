/*----------------------------------------------------
File:   alphabeta.c
Author: Dr. Paul M. Baggenstoss,
        NUWC, Newport, RI, code 21
        p.m.baggenstoss@ieee.org
        401-832-8240
Date:   Nov. 18, 1999
Subsequent Revisions:
----------------------------------------------------*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "mex.h"
#include "mexlib.h"

/* Input Arguments */
#define    L_IN prhs[0]
#define    S_IN prhs[1]
#define    hparm_IN prhs[2]
#define    Ahat_IN prhs[3]

/* Output Arguments */
#define    alphas_OUT    plhs[0]
#define    alognorm_OUT    plhs[1]
#define    betas_OUT    plhs[2]
#define    blognorm_OUT    plhs[3]
#define    gammaN_OUT    plhs[4]
#define    gamma_OUT    plhs[5]
#define    Ahat_OUT    plhs[6]


#define MAXN 100
#define MAXNPDF 1000
#define MAXFILLER 700
#define MAXPARTITION 100  /* maximum # of partitions for a given state */

#define DEBUG 0
#define DEBUG1 0
#define ALLOC_PSI 0

#define ALPHABETAMAX 0 /* if 1, alphas and betas have max of 1, otherwise, sum to 1 */

void pmbError(char * msg) ;
void * test_calloc(  int n, int sz) ;
/*
function [alphas,alognorm,betas,blognorm,gammaN,gamm,Ahat] = 
    alphabetacompress(Lin,S,hparm,ksegment,<Ahat>);
*/
void mexFunction( int  nlhs, MATRIX *plhs[], int  nrhs, const MATRIX *prhs[])
{

    double   *Lin, *S, ltmp, ltmp2, *lmax, penalties[MAXN][MAXPARTITION];
    double   psisum2, penalties2[MAXN][MAXPARTITION];
    double   *entry;
    int jcount2;
    int t, Npdf, icount, jcount, SDIM, p, l, n, idx,maxnpartition,nclass,Ntp;
    int Knext,N, ift, K[MAXN][MAXPARTITION], Nt, nsamp,i,iclass,P[MAXN], j, k, ii[MAXPARTITION];
    int S0[MAXN*MAXPARTITION];
    int first_wait[MAXN][MAXPARTITION];
    int partition_ptr_A[MAXN][MAXPARTITION];
    int do_train,do_A,last_wait[MAXN][MAXPARTITION];
    mxArray *hparm;
    mxArray *parm;
    double *ksegment,*bout,*aout, *tmp, *state_to_class_index, *pdf_to_class_index;
    double *alphas, *betas, *alognorm, *blognorm, max_l,max1,max2, *Pi, *A, *gamma, *gammaN, *Ahat_in, *Ahat_out;
    double psisum,ptmp,*psi,gammasum,*partition_distrib, *beta_end;


    if (nrhs < 3 ) {
        mexPrintf("Usage: [alphas,alognorm,betas,blognorm,gammaN,gamma,Ahat] = alphabetacompress(Lin,S,hparm,[Ahat]);\n");
    }
 

    hparm=(mxArray *) hparm_IN;

    parm = mxGetField( hparm , 0, "N"); N = *( (double *)  mxGetPr( parm) );
    if(parm ==0) pmbError("problem getting field N");

    if(N>MAXN) pmbError("N to big");

    parm = mxGetField( hparm , 0, "state_to_class_index"); 
    if(parm ==0) pmbError("problem getting field state_to_class_index,");
    state_to_class_index =  (double *)  mxGetPr( parm) ;

    parm = mxGetField( hparm , 0, "pdf_to_class_index"); 
    if(parm ==0) pmbError("problem getting field pdf_to_class_index");
    Npdf = mxGetNumberOfElements(parm); 
    if(Npdf>MAXNPDF) {
        mexPrintf("Npdf=%d\n",Npdf);
        pmbError("Npdf>MAXNPDF");
    }
    pdf_to_class_index =  (double *)  mxGetPr( parm) ;

    parm = mxGetField( hparm , 0, "pdf_entry"); 
    if(parm ==0) pmbError("problem getting field pdf_entry");
    entry =  (double *)  mxGetPr( parm) ;
    if(Npdf!= mxGetNumberOfElements(parm)) pmbError("Npdf!=length(entry)");

    parm = mxGetField( hparm , 0, "Pi"); 
    if(parm ==0) pmbError("problem getting field Pi");
    Pi =  (double *)  mxGetPr( parm) ;
    if(N!= mxGetNumberOfElements(parm)) pmbError("N!=length(Pi)");

    parm = mxGetField( hparm , 0, "A"); 
    if(parm ==0) pmbError("problem getting field A");
    if(mxGetM(parm)!=N || mxGetN(parm)!=N) pmbError("A not N by N");
    A =  (double *)  mxGetPr( parm) ;

    parm = mxGetField( hparm , 0, "ksegment"); 
    if(parm ==0) pmbError("problem getting field ksegment");
    if(mxGetNumberOfElements(parm) != Npdf) pmbError("mxGetNumberOfElements(hparm.ksegment) != Npdf");
    ksegment =  (double *)  mxGetPr( parm) ;

    parm = mxGetField( hparm , 0, "beta_end"); 
    if(parm ==0) {
         pmbError("problem getting field beta_end");
    } else {
       if(mxGetNumberOfElements(parm) != N) pmbError("mxGetNumberOfElements(hparm.beta_end) != N");
       beta_end = mxGetPr(parm);
    }



    CHECK_REAL_MATRIX(L_IN);
    CHECK_REAL_MATRIX(S_IN);


    Lin = mxGetPr(L_IN);
    if(mxGetM(S_IN) != N) pmbError(" mxGetM(S_IN) != N");
    nsamp = mxGetM(L_IN);
    if(mxGetN(L_IN) != Npdf) pmbError(" mxGetN(L_IN) != Npdf");

    SDIM=mxGetN(S_IN);
    S = mxGetPr(S_IN);

    /* count partitions , etc */
    maxnpartition=0;
    nclass=0;  
    Ntp=0;  /* counts total partitions */
    for (Nt=i=0; i<N; ++i) {
        iclass=state_to_class_index[i];
        nclass=MAX(nclass,iclass);
        for (k=j=0; j<Npdf; ++j) if(pdf_to_class_index[j]==iclass) ii[k++] = j;
        maxnpartition=MAX(maxnpartition,k);
        P[i] = k;
        /*mexPrintf("state %d is class %d and has %d partitions\n",i,iclass,P[i]); */
        if(P[i]>MAXPARTITION) { mexPrintf("P[i]>MAXPARTITION\n"); pmbError("P[i]>MAXPARTITION");}
        for (j=0; j<k; ++j) { 
           /*
            ift = pdf_to_featureset_index[ii[j]];
            K[i][j] = segment_size[ift-1]/base_shft;
           */
           K[i][j] = ksegment[ii[j]];
           Nt +=K[i][j];
           partition_ptr_A[i][j]=Ntp;
           Ntp++;
        }
    }
    if(SDIM != maxnpartition) pmbError("SDIM != maxnpartition");

    
    parm = mxGetField( hparm , 0, "PartitionDistrib"); 
    if(parm ==0) {
         pmbError("problem getting field PartitionDistrib");
    } else {
        if(mxGetM(parm) != maxnpartition) pmbError(" mxGetM(hparm.PartitionDistrib) != maxnpartition");
        if(mxGetN(parm) != N) pmbError(" mxGetN(hparm.PartitionDistrib) != N");
        partition_distrib = mxGetPr(parm);
    }

    if(nrhs <4 || mxGetNumberOfElements(Ahat_IN) ==0) {
      do_A=0;
    } else {
      if(mxGetM(Ahat_IN)!=Ntp || mxGetN(Ahat_IN)!=Ntp ) {
              mexPrintf("Ntp=%d\n",Ntp); 
              pmbError("Ahat_in must be Ntp by Ntp");
      }
      Ahat_in = mxGetPr(Ahat_IN);
      do_A=1;
    }

    /* computing Ahat_out requires beta, gamma (full training)  */
    if(nlhs>2) {
       do_train=1;
    } else {
       do_train=0;
       do_A=0;
    }
  
  /*determine partition transition weights */
  for (i=0; i<N; ++i) {
        iclass=state_to_class_index[i];
        for (k=j=0; j<Npdf; ++j) if(pdf_to_class_index[j]==iclass) ii[k++] = j;
        psisum=0;
        psisum2=0;
        for (j=0; j<k; ++j) { 
           if (partition_distrib != NULL ) {
              penalties[i][j]=partition_distrib[j+i*maxnpartition];
           } else  {
              if(j>0) {
                Knext=K[i][j-1];
              } else {
                Knext=2*K[i][j];
              }
              penalties[i][j]=pow(A[i+i*N],K[i][j]);
           }
           psisum += penalties[i][j];
           if(entry[ii[j]] > 0.0) {
              penalties2[i][j]=penalties[i][j];
              psisum2 += penalties2[i][j];
           } else {
              penalties2[i][j]=0.0;
           }
        }
        for (j=0; j<k; ++j) penalties[i][j] /= psisum;
        for (j=0; j<k; ++j) penalties2[i][j] /= psisum2;
  }
    
 
    /*mexPrintf("N=%d, nsamp=%d, base_shft=%d, Nt=%d, Npdf=%d\n",N,nsamp, base_shft, Nt, Npdf); */

    alphas_OUT = CREATE_DOUBLE(nsamp,Nt); alphas=mxGetPr(alphas_OUT);
    betas_OUT = CREATE_DOUBLE(nsamp,Nt); betas=mxGetPr(betas_OUT);
    alognorm_OUT = CREATE_DOUBLE(nsamp,1); alognorm=mxGetPr(alognorm_OUT);
    blognorm_OUT = CREATE_DOUBLE(nsamp,1); blognorm=mxGetPr(blognorm_OUT);
    gamma_OUT = CREATE_DOUBLE(nsamp,Nt); gamma=mxGetPr(gamma_OUT);
    gammaN_OUT = CREATE_DOUBLE(nsamp,N); gammaN=mxGetPr(gammaN_OUT);
    Ahat_OUT = CREATE_DOUBLE(Ntp,Ntp); Ahat_out=mxGetPr(Ahat_OUT);


   if(DEBUG1) mexPrintf("Trying to allocate aout, n=%d\n",Nt*sizeof(double));
   aout=(double *) test_calloc( Nt,sizeof(double));
   if(DEBUG1) mexPrintf("Trying to allocate tmp, n=%d\n",Nt*sizeof(double));
   tmp=(double *) test_calloc( Nt,sizeof(double));
   if(DEBUG1) mexPrintf("Trying to allocate lmax, n=%d\n",nsamp*sizeof(double));
   lmax=(double *) test_calloc( nsamp,sizeof(double));

   for(i=0; i<Nt; ++i) alphas[i*nsamp]=0;
   for(max_l=max1= -DBL_MAX,icount=0,i=0; i<N; ++i) {
     for(j=0; j<P[i]; ++j) {
              S0[i+j*N] = (int) ((S[i+j*N]-1)*nsamp);
              if(S0[i+j*N]<0|| S0[i+j*N]>nsamp*Npdf) {
                      mexPrintf("i=%d, j=%d, N=%d\n", i,j,N);
                      pmbError(" Oops S0");
              }
              ltmp = Lin[S0[i+j*N]];
              if(ltmp > -DBL_MAX && Pi[i] * penalties2[i][j] > 0) {
                ltmp2 = log(Pi[i] * penalties2[i][j]) + ltmp;
              } else {
                ltmp2 =  -DBL_MAX; 
              }
              max1=MAX(max1,ltmp2);
              max_l=MAX(max_l,ltmp);
              alphas[icount*nsamp]  = ltmp2;
      /*mexPrintf("(%d) %e\n",icount,ltmp2); */
              icount += K[i][j];
     }
     /* mexPrintf("\n"); */
   }

   max2=0;
   for(icount=0,i=0; i<N; ++i) {
     for(j=0; j<P[i]; ++j) {
              alphas[icount*nsamp]  = exp( alphas[icount*nsamp] - max1);
              max2 += alphas[icount*nsamp];
              icount += K[i][j];
     }
   }
   #if(ALPHABETAMAX)
     max2=1;
   #else
      for(icount=0,i=0; i<N; ++i) {
        for(j=0; j<P[i]; ++j) {
              alphas[icount*nsamp] /= max2;
              icount += K[i][j];
        }
      }
   #endif

   lmax[0]=max_l;
   alognorm[0]=max1+log(max2);


   icount=0;
   for(i=0; i<N; ++i) { 
        for(p=0; p<P[i]; ++p) {  
           first_wait[i][p] = icount;
           last_wait[i][p]=first_wait[i][p]+K[i][p]-1;
           icount += K[i][p];
        }
   }

   for(t=1; t<nsamp; ++t) {
         for(i=0; i<Nt; ++i) tmp[i]=0;
         jcount=0;
         for(j=0; j<N; ++j)  {
             jcount2=jcount;
             for (l=0; l<P[j]; ++l) {
                for(i=0; i<N; ++i) { 
                    for(p=0; p<P[i]; ++p) {  
                       if(i!=j) {
                            /* last wait state of state i partition p, to  */
                            /* first wait of state j, partition l, i!= j */
                            if( penalties2[j][l]  > 0.0) {
                              icount = last_wait[i][p];
                              tmp[jcount2] += alphas[t-1 + icount*nsamp] * A[i+j*N] * penalties2[j][l];
                            } 
                       }
                    }
                }
                jcount2 += K[j][l];
             }

             for (l=0; l<P[j]; ++l) {
                jcount +=1;
                for (n=1; n<K[j][l]; ++n) {
                    /* wait state increment */
                    icount = first_wait[j][l]+n-1;
                    tmp[jcount] += alphas[t-1 + icount*nsamp];
                    jcount=jcount+1;
                }
                jcount -=  K[j][l];
                for (p=0; p<P[j]; ++p) {
                     /* last wait state of state j, partition p  to first wait of state j partition l */
                     icount = last_wait[j][p];
                     tmp[jcount] +=  alphas[t-1 + icount*nsamp] * A[j+j*N] * penalties[j][l];
                }
                jcount += K[j][l];
             }
         }

         for(i=0; i<Nt; ++i) aout[i]=0;
         jcount=0;
         max_l = max1= -DBL_MAX;
         for(j=0; j<N; ++j) { 
             for (l=0; l<P[j]; ++l) {
                for (n=0; n<K[j][l]; ++n) {
                   if(t>=n) {
                     idx=t-n;
                     ltmp = Lin[idx+S0[j+l*N] ];
                     if(ltmp > -DBL_MAX && tmp[jcount] > 0) {
                       ltmp2=log(tmp[jcount]) + ltmp;
                     } else {
                       ltmp2= -DBL_MAX;
                     }
                     max1=MAX(max1,ltmp2);
                     max_l=MAX(max_l,ltmp);
                     aout[jcount]=ltmp2;
                   }
                   ++jcount;
               }
            }
        }

        jcount=0;
        for(j=0; j<N; ++j) { 
            for (l=0; l<P[j]; ++l) {
               for (n=0; n<K[j][l]; ++n) {
                  if(t>=n) {
                    aout[jcount]=exp( aout[jcount] - max1);
                  }
                   ++jcount;
               }
            }
        }

#if(DEBUG)
        /*if(t==2) for(i=0; i<Nt; ++i) mexPrintf("tmp aout [%d]=[%f %f]\n",i,tmp[i],aout[i]); */
#endif

         #if(ALPHABETAMAX)
            for (max2= -DBL_MAX, i=0; i<Nt; ++i) max2=MAX(max2,aout[i]);
         #else
            for (max2=0, i=0; i<Nt; ++i) max2+=aout[i];
         #endif
         for (i=0; i<Nt; ++i) aout[i] /= max2;


        lmax[t]=max_l;
        alognorm[t] = max1+log(max2) + alognorm[t-1];

         /*alphas(t,:) = aout'; */
         for (i=0; i<Nt; ++i) alphas[t+i*nsamp] = aout[i];

   } /* for t */

   if(!do_train)  {
         free(lmax);
         free(tmp);
         free(aout);
         return; 
   }

   if(DEBUG1) mexPrintf("Trying to allocate bout, n=%d\n",Nt*sizeof(double));
   bout=(double *) test_calloc( Nt,sizeof(double));
   if(beta_end==NULL) {
       for (i=0; i<Nt; ++i) bout[i]=1;
   } else {
       /* Implement custom end probabilities - to insure that the state not end in some states 
       */
       for (l=i=0; i<N; ++i) {
           for (j=0; j<P[i]; ++j) { 
              for (k=0; k<K[i][j]; ++k) bout[l++]=beta_end[i];
           }
       }
   }
  #if(ALPHABETAMAX)
     blognorm[nsamp-1]=0;
  #else
      for (max2=0, i=0; i<Nt; ++i) max2+=bout[i];
      for (i=0; i<Nt; ++i) bout[i] /= max2;
      blognorm[nsamp-1]=log(max2);
  #endif

   for (i=0; i<Nt; ++i) betas[(nsamp-1)+i*nsamp]=bout[i];

   for(t=nsamp-2; t>=0; --t) {

      jcount=0;
      max1= -DBL_MAX;
      for(j=0; j<N; ++j) { 
             for (l=0; l<P[j]; ++l) {
                for (n=0; n<K[j][l]; ++n) {
                   idx=(t-n+1+nsamp)%nsamp;
                   ltmp = Lin[idx+S0[j+l*N]];
                   if(ltmp > -DBL_MAX && betas[t+1+jcount*nsamp] > 0) {
                          ltmp2 = log(betas[t+1+jcount*nsamp])+ltmp;
                          max1=MAX(max1,ltmp2);
                   } else {
                          ltmp2 = -DBL_MAX;
                   }
                   tmp[jcount]=ltmp2;
                   ++jcount;
               }
            }
      }

      jcount=0;
      for(j=0; j<N; ++j) { 
             for (l=0; l<P[j]; ++l) {
                for (n=0; n<K[j][l]; ++n) {
                   idx=(t-n+1+nsamp)%nsamp;
                   tmp[jcount]=exp(tmp[jcount] - max1);
                   ++jcount;
               }
            }
      }

   for (i=0; i<Nt; ++i) bout[i]=0;
      jcount=0;
      for(j=0; j<N; ++j)  {
          jcount2=jcount;
          for(l=0; l<P[j]; ++l) {
             for(i=0; i<N; ++i) { 
                 for(p=0; p<P[i]; ++p) {
                         if(i!=j ) {
                              /* last wait state of state i partition p, to first partition,  */
                              /* first wait of state j, i!= j */
                              icount = last_wait[i][p];
                              bout[icount] += tmp[jcount2] * A[i+j*N] * penalties2[j][l];
                         }
                         icount += 1;
                 }
             }
             jcount2 += K[j][l];
          }

             for (l=0; l<P[j]; ++l) {
                jcount +=1;
                for (n=1; n<K[j][l]; ++n) {
                    /* wait state increment */
                    icount = first_wait[j][l]+n-1;
                    bout[icount] += tmp[jcount];
                    jcount=jcount+1;
                }
                jcount -=  K[j][l];

                for (p=0; p<P[j]; ++p) {
                     /* last wait state of state j, partition p  to first wait of state j partition l */
                     icount = last_wait[j][p];
                     bout[icount] +=  tmp[jcount] * A[j+j*N] * penalties[j][l];
                }
                jcount += K[j][l];
             }
      }



  #if(ALPHABETAMAX)
      /* max2=max(bout); */
      for (max2= -DBL_MAX, i=0; i<Nt; ++i) max2=MAX(max2,bout[i]);
      /* bout = bout / max2; */
      for (i=0; i<Nt; ++i) bout[i] /= max2;
  #else
      for (max2=0, i=0; i<Nt; ++i) max2+=bout[i];
      for (i=0; i<Nt; ++i) bout[i] /= max2;
  #endif

      /* betas(t,:)=bout'; */
       for (i=0; i<Nt; ++i) betas[t+i*nsamp] = bout[i];

      /* blognorm(t) = log(max2) + blognorm(t+1) + max1; */
        blognorm[t] = max1+log(max2) + blognorm[t+1];

   } /* for t */

   /* Compute gamma */
   for(t=0; t<nsamp; ++t) {
         for(gammasum=i=0; i<Nt; ++i) {
            gamma[t+i*nsamp]=alphas[i*nsamp+t]*betas[i*nsamp+t];
            gammasum += gamma[t+i*nsamp];
        }
        for(i=0; i<Nt; ++i) gamma[t+i*nsamp] /= gammasum;

        for(i=0; i<N; ++i)  {
            gammasum=0;
            for(p=0; p<P[i]; ++p) for(n=0; n<K[i][p]; ++n) {
                gammaN[t+i*nsamp] += gamma[t+(first_wait[i][p]+n)*nsamp];
            }
        }
   }

   if(do_A) {
     if(DEBUG1) mexPrintf("Trying to allocate psi, n=%d\n",Ntp*Ntp*sizeof(double));
     psi=test_calloc(Ntp*Ntp,sizeof(double));

     /* Accumulate state transitions */
     for(i=0; i<Ntp*Ntp; ++i) Ahat_out[i]  = Ahat_in[i];
     for(t=0; t<nsamp-1; ++t) {
         psisum=0;
         for(i=0; i<Ntp*Ntp; ++i) psi[i]=0;
         for(i=0; i<N; ++i) for(p=0; p<P[i]; ++p) {
             for(j=0; j<N; ++j) for(l=0; l<P[j]; ++l) {
                 /* last wait state of state i, partition p  to first wait of state j partition l */
                 icount = t + last_wait[i][p]*nsamp;
                 jcount = t+1 + first_wait[j][l]*nsamp;
                 ptmp =  A[i+j*N] * penalties2[j][l] * exp(Lin[t+1+S0[j+l*N]] -lmax[t+1]);
                 ptmp *=  betas[jcount] * alphas[icount];
                 psisum += ptmp;
                 psi[partition_ptr_A[i][p]+partition_ptr_A[j][l]*Ntp] += ptmp;
             }
             /* within-partition transitions */
             icount = t + first_wait[i][p]*nsamp;
             jcount = t+1 + first_wait[i][p]*nsamp;
             max2 =  exp(Lin[t+S0[i+p*N]] -lmax[t+1]);
             for(l=0; l<K[i][p]-1; ++l) {
                    /* wait state l to l+1 */
                    ptmp =  max2 * alphas[icount+l] * betas[jcount+l+1];
                    psisum += ptmp;
             }
         }
         if(psisum>0.0) { 
            for(i=0; i<Ntp*Ntp; ++i) Ahat_out[i] += psi[i]/psisum;
         } else {
           /*mexPrintf("warning: t=%d: psisum is zero\n",t); */
         }
      }
      free(psi);
   }

   free(bout);
   free(lmax);
   free(tmp);
   free(aout);
}



void pmbError(char * msg) {
    mexPrintf("%s\n",msg);
    mexErrMsgTxt(msg);
} 

void * test_calloc(  int n, int sz) {
   void * ptr;
   ptr = calloc( n,sz);
   if(ptr==NULL) {
          mexPrintf("Cannot allocate %d\n",n);
          mexErrMsgTxt("out");
   }
   return ( ptr);
}
