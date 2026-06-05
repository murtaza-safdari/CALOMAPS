 /* pixel32.f -- translated by f2c (version 20000817).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Fully vectorize the charge deposition and include the effect of the magnetic field on the delta-rays */
/* Use Henrich mobility and Llubjana trapping */
/* Reduce Hall factor to 1.12 (12/10/03) */
/* Adaptive step sizing using Cash-Karp embedded 5th-order technique: version (6/23/05) */
/* Add header output to barrel_ten.out.  Switched on by adding the character '1' to the */
/* beginning of the ascii header in pixel.init (7/05/05) */
/* Add pion energy dependence of cross sections from H. Bichsel.  Use magnitude of pion direction to */
/* store the information.  Assumes 45 GeV if not specified in old files (11/10/05) */
/* Change pixel array to 21x7 to accommodate wider range of input angles (04/10/06) */
/* Version to automatically generate multiple output files while incrementing the cluster length */
/* Add electron hall factor rhe to input list */
/* Add NIST Estar inverse stopping powers: drde (11/15/2007) */
/* Add multiple scattering to primary delta rays for NIST Estar choice (11/14/2009) */
/* Increase event array size to 21x13 (11/21/2008) */
/* Read cot(alpha), cot(beta) and ppion from an external file */
/* Multiple run version for use on cluster */
/* Enforce efield boundary conditions */
/* Fix factor of 2 in diffusion equation */
/* Version that incorporates the temporal response of the preamp */
/* Print current from largest pixels */
/* Include external weighting potential lookup table in wgt_pot.init */
/* Randomize impact point over a 3x3 pixel array */
/* Pass through module impact y and track pT */
/* Fix event to event momentum dependence of charge deposition (07/18/2022) */


#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stddef.h>
#ifdef __POWERPC__
#include <altivec.h>
#undef pixel
#else
#if defined(__arm64__) || defined(__ARM_NEON)
#include "sse2neon.h"
#else
#include <xmmintrin.h>
#endif
#endif

/* Global symbols */

/* Maximum number of e-h pairs to be stored in static arrays */

#define NEHSTORE 150000

/* Define pixel signal buffer sizes */

#define TXSIZE 21
#define TYSIZE 13

/* Define maximum E-field and weighting potential array sizes */

#define NARRAYX 26
#define NARRAYY 13
#define NARRAYZ 51


/* Define the maxumum number of runs */

#define TEMPMAX 500

/* Define track list size */

#define NMUON  50000

/* Define the number of steps in the crrc response function */

#define NCRRC 20

/* Table of constant values */

static int c__1 = 1;
static int c__2 = 2;
static int c__3 = 3;
static int c__4 = 4;
static int c__120 = 120;

/* Prescaling factor for electrons and holes: transport only 1/Nscale carriers to save time */

static int Nscale = 1;  /* This doesn't cause additional fluctuations (we already get 22,000 e-h pairs per 300um Si) */

   typedef union vect_or_f {
#ifdef __POWERPC__
     vector float v;
#else
     __m128 v;
#endif
     float f[4];
   } vect_or_f;
#ifdef __POWERPC__   
   typedef union vect_or_c {
     vector unsigned char v;
     unsigned char c[16];
   } vect_or_c;
#endif
   typedef union vect_or_i {
#ifdef __POWERPC__   
     vector unsigned int v;
#else
  __m128 v;
#endif
     unsigned int i[4];
   } vect_or_i;

/* Define the E-field and weighting potential arrays and associated quantities */

    static vect_or_f efield[NARRAYX][NARRAYY][NARRAYZ];
    static vect_or_f wgtpot[NARRAYX][NARRAYY][NARRAYZ][3];
    static int npixx, npixy, npixz;    
/* Define array to help with bounds checking */
    static int mnode[3];
    static vect_or_f bfield;
    static float bfield_z;
    static char header[80];
    
/* function prototypes */

#include "ppixelav2_prototypes.h"

/* Main program */ int main(int argc, char *argv[])
{
    /* System generated locals */
    int i__1;
    float r__1, r__2;
    double d__1, d__2;

    /* Local variables */
    static float vect[6];
    static float cotatrack[NMUON], cotbtrack[NMUON], ppiontrack[NMUON], modxtrack[NMUON], modytrack[NMUON], pttrack[NMUON];
    static int flipped[NMUON];
    static float thick, xsize, ysize, temp, flux[2], rhe, rhh, peaktim, samptim, stimstp;    
    static int i__, indeh[2][NEHSTORE]	/* was [2][300000] */;
    static int nto2in, lux, initseed, ivec[25];
    static float pixel[TXSIZE][TYSIZE];
    static float pixhist[NCRRC+1][TXSIZE][TYSIZE];
    static float crrcresp[NCRRC];
    static int ntotin, neh, ntrack;
    static vect_or_f xeh[2][NEHSTORE]	/* was [4][2][300000] */;
    static vect_or_f xhisteh[NCRRC][2][NEHSTORE];
    time_t now;
    struct tm *nows;
    int sec, min, hour, yday, j, k;
    static int fileind, filebase, fileoff, runsize, irun, ievent, frun, nskip, procid, new_drde, ehole;
    static float rvec[4], pimom, xoffset, yoffset, lenxmin, lenxmax, deltaxlen, lenymin, lenymax, deltaylen, locdir[3], cotalpha, cotbeta;
    static float clusxlen, clusylen;
    static char outfile[500], seedfile[500];
    static double alpha;

    FILE *isfp, *iifp, *ofp, *icfp;

    char track_list[500] = "track_list.txt";
    
    /* If no arguments, quit */
	
    if(argc < 2) {
      printf("Need at least one argument to specify run \n");
      return 0;
    }
	
    /* A single argument is a first run number  */
    
    if(argc == 2) {
      sscanf(argv[1],"%d", &frun);
      if(frun < 1 || frun > TEMPMAX) {printf("frun %d is illegal, quit \n", frun); return 0;}
      runsize = 30000;
      printf("Skipping %d blocks of runsize %d \n", frun-1, runsize);
    }

    /* 2 arguments = first run number, second track list */
    if(argc == 5) {
      sscanf(argv[1],"%d", &frun);
      if(frun < 1 || frun > TEMPMAX) {printf("frun %d is illegal, quit \n", frun); return 0;}
      runsize = 30000;
      printf("Skipping %d blocks of runsize %d \n", frun-1, runsize);
      sscanf(argv[2],"%s", &track_list);
      sscanf(argv[3],"%s", &outfile);
      sscanf(argv[4],"%s", &seedfile);
      printf("Track list file: %s \n", track_list);
    }
    
    /* /\* If two arguments, second could be a number of runs or a fork instruction *\/ */
	
    /* if(argc == 3) { */
    /*   sscanf(argv[1],"%d", &frun); */
    /*   if(frun < 1 || frun > TEMPMAX) {printf("frun %d is illegal, quit \n", frun); return 0;} */
    /*   if(*argv[2] == 'f') {runsize = 30000;} */
    /*   else { */
    /* 	sscanf(argv[2],"%d", &runsize); */
    /* 	if(runsize < 1 || runsize > NMUON) {printf("runsize %d is illegal, quit \n", runsize); return 0;} */
    /*   }	 */
    /*   printf("Skipping %d blocks of runsize %d \n", frun-1, runsize); */
    /*   if(*argv[2] == 'f') { */
    /* 	procid = fork(); */
    /* 	if(procid) { */
    /* 	  printf("Forking process, id = %d\n", procid); */
    /* 	  return 0;  */
    /* 	}			 */
    /*   } */
    /* } */
	
    /* /\* If three arguments, retrieve first run, number of runs, and possible fork command *\/ */
    
    /* if(argc == 4) { */
    /*   sscanf(argv[1],"%d", &frun); */
    /*   if(frun < 1 || frun > TEMPMAX) {printf("frun %d is illegal, quit \n", frun); return 0;} */
    /*   sscanf(argv[2],"%d", &runsize); */
    /*   if(runsize < 1 || runsize > NMUON) {printf("runsize %d is illegal, quit \n", runsize); return 0;} */
    /*   printf("Skipping %d blocks of runsize %d \n", frun-1, runsize); */
    /*   if(*argv[3] == 'f') { */
    /* 	procid = fork(); */
    /* 	if(procid) { */
    /* 	  printf("Forking process, id = %d\n", procid); */
    /* 	  return 0;  */
    /* 	}			 */
    /*   } */
    /* } */
	
    /*  Define the detector parameters from the global initialization file */

    pixinit(&pimom, &thick, &xsize, &ysize, &temp, flux, &rhe, &rhh, &peaktim, &samptim, &ehole, &new_drde, &filebase);

    crrc(peaktim, samptim, &stimstp, crrcresp);
    printf("sample time %f \n", stimstp);
    for(j=0; j<NCRRC; ++j) {
      printf("crrcresp[%d] = %f \n", j, crrcresp[j]);
    }
	
    /*  read track list */
    icfp = fopen(track_list, "r");
    if (icfp==NULL) {
      printf("no track_list.txt file found/n");
      return 0;
    }
	
    /* Skip to the right place */
	
    nskip = runsize*(frun-1);
	
    if(nskip > 0) {
      
      ntrack = 0;
      while(fscanf(icfp,"%f %f %f %d %f %f %f", &cotatrack[0], &cotbtrack[0], &ppiontrack[0], &flipped[0], &modxtrack[0], &modytrack[0], &pttrack[0]) != EOF) {
	++ntrack; 
	if(ntrack >= nskip) break;
      }		
    }

    /* Now read-in track angles and momenta to process */
	
    ntrack = 0;
    while(fscanf(icfp,"%f %f %f %d %f %f %f", &cotatrack[ntrack], &cotbtrack[ntrack], &ppiontrack[ntrack], &flipped[ntrack], &modxtrack[ntrack], &modytrack[ntrack], &pttrack[ntrack]) != EOF) {
      ++ntrack;
      if(ntrack == NMUON) break;
      if(ntrack >= runsize) break;
    }
    fclose(icfp);
    printf("number of tracks = %d \n", ntrack);
	
    /* the file index is the sum of an overall base number and a local run offset */
    
    fileind = filebase + frun;
    
    /*  Create a seedfile name for this run */
    
    // sprintf(seedfile,"seedfile%5.5d",fileind);
	
    /*  Determine current time */

    now = time(NULL);
    nows = localtime(&now);
    sec = (*nows).tm_sec;
    min = (*nows).tm_min;
    hour = (*nows).tm_hour;
    yday = (*nows).tm_yday;
    
    printf("Begin on day %d at %02d:%02d:%02d\n", yday, hour, min, sec);

    /*  Initialize the random number generation */

    /* First check to see if any intermediate state has been saved */

    isfp = fopen(seedfile, "r");
    if (isfp==NULL) {

      /* If no seedfiled, use single seed and set-up generator */

      lux = 3;
      ntotin = 0;
      nto2in = 0;
      initseed = fileind;
      rluxgo_(&lux, &initseed, &ntotin, &nto2in);
      irun=0; ievent=0;
    } else {
    
      /* read-in 25 ints and restore generator to previous state */

      fscanf(isfp,
	     "%x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %d %d", 
	     &ivec[0], &ivec[1], &ivec[2], &ivec[3], &ivec[4], &ivec[5], &ivec[6], &ivec[7],
	     &ivec[8], &ivec[9], &ivec[10], &ivec[11], &ivec[12], &ivec[13], &ivec[14], &ivec[15],
	     &ivec[16], &ivec[17], &ivec[18], &ivec[19], &ivec[20], &ivec[21], &ivec[22], &ivec[23],
	     &ivec[24], &irun, &ievent);
      fclose(isfp);
      rluxin_(ivec);
    }
    
    /*  Create a filename for this run */
    // sprintf(outfile,"pixel_clusters_d%5.5d.out",fileind);
    
    if(ievent==0 && frun == 1) {
		
      /*  copy  header to the output file on first event */
      
      ofp = fopen(outfile, "w");
      fprintf(ofp,"%s \n", &header[0]);
      fprintf(ofp,"%f  %f  %f  %f\n", xsize, ysize, thick, stimstp);
      fclose(ofp); 	   
    }
    
    
    while(ievent < ntrack) {

      /* Generate initial position and direction of the track */
      
      ranlux_(rvec,&c__4);
      cotbeta = cotbtrack[ievent];	
      if(fabsf(cotbeta) > 10.) goto incr;
      cotalpha = cotatrack[ievent];
      if(fabsf(cotalpha) > 10.) goto incr;
      locdir[2] = 1./sqrt((double)(1.+cotbeta*cotbeta+cotalpha*cotalpha));
	/* track travels in the E-field direction in the unflipped coordinate system */
      if(flipped[ievent] == 0) locdir[2] = -locdir[2];
      locdir[0] = cotbeta*locdir[2];
      locdir[1] = cotalpha*locdir[2];
			   
      /*  Calculate the offsets from the detector center to its front face */
      
      xoffset = locdir[0]/locdir[2] * thick / 2.;
      yoffset = locdir[1]/locdir[2] * thick / 2.;
      
      if(locdir[2] < 0.) {
	vect[2] = thick;
      } else {
	vect[2] = 0.;
      }
      
      /* ===== CALOMAPS REAL-ENTRY PATCH =====================================================
         Upstream draws the sub-pixel midplane impact at random over the central 3x3 pixels
         (3.*xsize*(rvec-0.5)); its purpose there is to SCAN position dependence into a template.
         For CALOMAPS we instead inject the real per-crossing impact carried in modx/mody (the
         deck columns), so PIXELAV simulates the cluster at the track's actual sub-pixel position.
         modx/mody arrive in microns full-truth (modx = our v -> x/21px ; mody = our u ->
         y/13px Lorentz); we reduce them mod-pitch to a [-pitch/2,+pitch/2] sub-pixel midplane
         offset, then back-project to the entry face exactly as upstream does. rvec[0],rvec[1]
         are left unused (ranlux_ still called, so the downstream RNG stream is unchanged). */
      vect[0] = (modxtrack[ievent] - xsize*floorf(modxtrack[ievent]/xsize + 0.5f)) + (vect[2] - thick/2.)*locdir[0]/locdir[2];
      vect[1] = (modytrack[ievent] - ysize*floorf(modytrack[ievent]/ysize + 0.5f)) + (vect[2] - thick/2.)*locdir[1]/locdir[2];
      vect[3] = locdir[0]*ppiontrack[ievent];
      vect[4] = locdir[1]*ppiontrack[ievent];
      vect[5] = locdir[2]*ppiontrack[ievent];

      /*  Set Bfield z-direction for this event */
      
      bfield.f[2] = bfield_z;
      if(cotbeta < 0.) {bfield.f[2] = -bfield_z;}
      
      /*  Propagate the track and make e-h pairs */
      
      deposit(vect, thick, new_drde, NEHSTORE, xeh, &neh);
      
      /*  don't process overflows */
      
      if(neh < NEHSTORE) {
	
	/*  Propagate the e's and h's (no signal after the sample time) */
	
	propag(thick, xsize, ysize, temp, flux, rhe, rhh, samptim, stimstp, ehole, neh, xeh, indeh, xhisteh);
	
	/*  Count e's and h's on various pixels */
	
	detect(xsize, ysize, thick, ehole, xhisteh, neh, crrcresp, pixel, pixhist);
	
	/*  Write out the results to a file */
	
	ofp = fopen(outfile, "a");
	fprintf(ofp,"<cluster>\n");
	fprintf(ofp,
		"%f %f %f %f %f %f %d %f %f \n", 
		vect[0], vect[1], vect[2], vect[3], vect[4], vect[5], neh, modytrack[ievent], pttrack[ievent]);
	for(k = 1; k<=NCRRC; ++k) {
	  fprintf(ofp,"<time slice %f ps>\n", k*stimstp);
	  for (j = 0; j < TYSIZE; ++j) {
	    fprintf(ofp,
		    "%2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f\n", 
		    pixhist[k][0][j], pixhist[k][1][j], pixhist[k][2][j], pixhist[k][3][j], pixhist[k][4][j], 
		    pixhist[k][5][j], pixhist[k][6][j], pixhist[k][7][j], pixhist[k][8][j], pixhist[k][9][j],
		    pixhist[k][10][j], pixhist[k][11][j], pixhist[k][12][j], pixhist[k][13][j], pixhist[k][14][j],
		    pixhist[k][15][j], pixhist[k][16][j], pixhist[k][17][j], pixhist[k][18][j], pixhist[k][19][j], 
		    pixhist[k][20][j]);
	  }
	}    
	fclose(ofp);      
      } 
      
    incr: ievent += 1;
      
      /* Save current random number state */    
      
      rluxut_(ivec);
      isfp = fopen(seedfile, "w");
      fprintf(isfp,
	      "%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x%9x %d %d\n", 
	      ivec[0], ivec[1], ivec[2], ivec[3], ivec[4], ivec[5], ivec[6], ivec[7],
	      ivec[8], ivec[9], ivec[10], ivec[11], ivec[12], ivec[13], ivec[14], ivec[15],
	      ivec[16], ivec[17], ivec[18], ivec[19], ivec[20], ivec[21], ivec[22], ivec[23],
	      ivec[24], irun, ievent);
      fclose(isfp);
      
      /*  Determine current time */
      
      now = time(NULL);
      nows = localtime(&now);
      sec = (*nows).tm_sec;
      min = (*nows).tm_min;
      hour = (*nows).tm_hour;
      yday = (*nows).tm_yday;

      // if(ievent < 25) {printf("day %d at %02d:%02d:%02d, run %d, event %d, number of e-h pairs = %d\n", yday, hour, min, sec, irun, ievent, neh);}
      printf("day %d at %02d:%02d:%02d, run %d, event %d, number of e-h pairs = %d\n", yday, hour, min, sec, irun, ievent, neh);
    }
    
    irun += 1;
    ievent = 0;
    
    /*  Determine current time */
    
    now = time(NULL);
    nows = localtime(&now);
    sec = (*nows).tm_sec;
    min = (*nows).tm_min;
    hour = (*nows).tm_hour;
    yday = (*nows).tm_yday;
    
    printf("End on day %d at %02d:%02d:%02d\n", yday, hour, min, sec);
    
    
} /* MAIN__ */

#include "ppixelav2.c"


