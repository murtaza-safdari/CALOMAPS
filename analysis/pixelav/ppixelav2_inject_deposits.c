/* ppixelav2_inject_deposits.c -- PIXELAV drift and readout of given energy deposits.

   PIXELAV's list drivers make the e-h pairs with deposit(), which follows one straight track
   through the full sensor thickness. This driver takes the energy deposits from outside instead
   (straight segments with an energy, e.g. the steps of a Geant4 shower in the silicon), turns
   them into e-h pairs and runs PIXELAV's propag() (drift, diffusion, trapping) and detect()
   (induced and collected charge on the 21 x 13 pixel window) on them.

   Usage, in a PIXELAV run directory (ppixel2.init, wgt_pot.init, SIRUTH.SPR):
       ulimit -s unlimited          (macOS: ulimit -s 65520; propag() needs ~25 MB of stack)
       ppixelav2_inject_deposits <jobs_file> <out_file> [seed] [slices]
     seed    non-negative integer (default 1)
     slices  last (default): write the final time slice; all: write all NCRRC slices

   Frame and units: the window frame of detect() and of the list driver's tracks.
     x [um]  along the 21-pixel side (pitch xsize); y [um] along the 13-pixel side (pitch ysize).
             Pixel (10,6) is centred on (0,0): the window is |x| < 10.5 xsize, |y| < 6.5 ysize.
     z [um]  depth from 0 to thick, oriented as the list driver's tracks (a deck line with
             flipped=1 runs from z=0 to z=thick). Electrons drift against the E field of
             ppixel2.init; in the decal320 and bpix100_50x25 models E_z < 0, so with ehole=0 the
             electrons are collected at z=thick (the side pixinit() prints as "n-side").
     edep    [eV]

   Jobs file, repeated blocks (blank lines and lines starting with '#' are ignored):
       J <job_id> <n_seg>
       <x1> <y1> <z1> <x2> <y2> <z2> <edep_eV>        (n_seg lines)
     job_id >= 0 (64-bit), n_seg >= 0, edep_eV >= 0.

   Per job:
     - a segment with n = edep_eV / 3.68 makes floor(n) + Bernoulli(n - floor(n)) pairs. The
       deposit already carries Geant4's energy-loss fluctuation, so no further smearing is
       added (deposit(), which draws its own Poisson numbers per collision, is not called).
     - the pairs lie uniformly at random along the segment, each pair being one entry of
       xeh[0] (electron) and xeh[1] (hole) at the same point, time 0 (the layout deposit()
       leaves). z is clipped to [ZMARGIN, thick - ZMARGIN]: propag() and detect() take
       z <= 0 and z >= thick as having left the slab.
     - pairs created outside the 21 x 13 window are counted (n_out) and not transported: their
       charge would fall outside the window, except that detect() truncates (int) x/xsize+10.5
       and (int) y/ysize+6.5 towards zero and would put those just below the window's low
       x or y edge into column 0 or row 0.
     - every carrier is transported (Nscale = 1).
     - RANLUX is re-seeded from (seed, job_id) and triplg()'s buffer of Gaussian deviates is
       emptied first, so a job's cluster does not depend on the other jobs in the file or on
       their order.
   Not modelled: the spread of deposits inside a segment beyond a uniform line (the segments
   should be short: Geant4 steps, cut if long); the arrival times of the deposits (all pairs of
   a job start drifting at t = 0); any interaction between carriers (PIXELAV has none, so
   clusters add linearly and a job split in two gives the same charge in sum). The B field is
   taken from ppixel2.init as it is (the list drivers flip its sign with cot_beta).

   Refused with a line starting "ERROR" and exit status 1, before any transport: a malformed
   block, a non-finite number, edep < 0, a segment end farther outside the window than the
   window's own size (|x| > 31.5 xsize, |y| > 19.5 ysize, z < -thick, z > 2 thick), or a job
   whose pairs certainly reach NEHSTORE. During transport: a job whose drawn pairs reach
   NEHSTORE. Such jobs must be split (exact, by linearity); out_file is then incomplete.

   Output: line 1 the sensor title, line 2 "xsize ysize thick stimstp", then per job
       <cluster>
       <job_id> <neh> <n_seg> <edep_eV_total> <n_out>
       <time slice T ps>                 (the last slice, or all NCRRC of them)
       13 rows of 21 charges             (row j = pixel row y_j, column i = x_i; electrons)
   neh = pairs transported, neh + n_out = pairs created. The slices are cumulative: the last
   holds the charge at the end of the sampling time.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>
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

#define NEHSTORE 1000000

/* Define pixel signal buffer sizes */

#define TXSIZE 21
#define TYSIZE 13

/* Define maximum E-field and weighting potential array sizes */
/* NARRAYY is 26 here, 13 upstream: the 50x25 um model's grid is 26x26x51, the others are 26x13x51.
   A grid smaller than the arrays fills part of them; fieldsav/wgtpotsav index each axis only up
   to npix-1. */

#define NARRAYX 26
#define NARRAYY 26
#define NARRAYZ 51

/* Define the number of steps in the crrc response function */

#define NCRRC 20

/* Energy per e-h pair in eV (as in deposit()) */

#define EPAIR 3.68

/* Created pairs are kept this far (um) inside the faces z=0 and z=thick */

#define ZMARGIN 1.0e-3

/* rluxgo_() seeds its multiplicative congruential generator with 1 .. RLUXSEEDMAX */

#define RLUXSEEDMAX 2147483562ULL

/* propag() keeps three float/int [2][NEHSTORE] arrays on the stack */

#define STACKNEED (32UL*1024UL*1024UL)

#define LINESIZE 4096

/* Table of constant values */

static int c__1 = 1;
static int c__2 = 2;
static int c__3 = 3;
static int c__120 = 120;

/* Prescaling factor for electrons and holes: propag() transports only every Nscale-th pair and
   parks the others where detect() does not count them. 1 = every pair. */

static int Nscale = 1;

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

/* Jobs-file reading state, for the error messages */

static const char *jobs_path;
static long long jobs_line;

/* stdout and /dev/null, to silence rluxgo_()'s two lines per job */

static int stdout_fd = -1, null_fd = -1;

static double seconds_since(struct timespec *t0)
{
  struct timespec t1;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  return (double)(t1.tv_sec - t0->tv_sec) + 1.e-9*(double)(t1.tv_nsec - t0->tv_nsec);
}

/* The next line of the jobs file that is not blank and does not start with '#' (leading
   whitespace skipped), or NULL at the end of the file. */

static char *next_line(FILE *fp, char *buf)
{
  char *s;
  size_t n;
  while(fgets(buf, LINESIZE, fp) != NULL) {
    ++jobs_line;
    n = strlen(buf);
    if(n == LINESIZE-1 && buf[n-1] != '\n' && !feof(fp)) {
      printf("ERROR: %s line %lld is longer than %d characters \n", jobs_path, jobs_line, LINESIZE-2);
      exit(1);
    }
    for(s = buf; *s == ' ' || *s == '\t' || *s == '\r' || *s == '\n'; ++s) {}
    if(*s == '\0' || *s == '#') continue;
    return s;
  }
  if(ferror(fp)) {
    printf("ERROR: reading %s failed after line %lld \n", jobs_path, jobs_line);
    exit(1);
  }
  return NULL;
}

/* "J <job_id> <n_seg>" -> 1, anything else -> 0 */

static int parse_job(const char *s, long long *job_id, int *nseg)
{
  int nc = -1;
  if(sscanf(s, "J %lld %d %n", job_id, nseg, &nc) != 2 || nc < 0 || s[nc] != '\0') return 0;
  return *job_id >= 0 && *nseg >= 0;
}

/* "<x1> <y1> <z1> <x2> <y2> <z2> <edep_eV>" -> 1, anything else -> 0 */

static int parse_seg(const char *s, double v[7])
{
  int nc = -1;
  if(sscanf(s, "%lf %lf %lf %lf %lf %lf %lf %n", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &nc) != 7
     || nc < 0 || s[nc] != '\0') return 0;
  return 1;
}

/* NULL for a usable segment, else what is wrong with it */

static const char *seg_problem(const double v[7], float xsize, float ysize, float thick)
{
  int i;
  double xlim = 1.5*TXSIZE*xsize, ylim = 1.5*TYSIZE*ysize;
  for(i = 0; i < 7; ++i) {
    if(!isfinite(v[i])) return "a number is not finite";
  }
  if(v[6] < 0.) return "edep_eV is negative";
  for(i = 0; i < 4; i += 3) {
    if(fabs(v[i]) > xlim || fabs(v[i+1]) > ylim || v[i+2] < -thick || v[i+2] > 2.*thick)
      return "an end lies farther outside the window than the window's own size";
  }
  return NULL;
}

/* Next segment line of the job that started on line jline, checked; exits on any problem */

static void read_seg(FILE *fp, char *buf, long long job_id, long long jline, double v[7],
                     float xsize, float ysize, float thick)
{
  const char *s, *why;
  s = next_line(fp, buf);
  if(s == NULL) {
    printf("ERROR: job %lld (%s line %lld): the file ends before its last segment \n", job_id, jobs_path, jline);
    exit(1);
  }
  if(!parse_seg(s, v)) {
    printf("ERROR: job %lld: %s line %lld is not 'x1 y1 z1 x2 y2 z2 edep_eV' \n", job_id, jobs_path, jobs_line);
    exit(1);
  }
  why = seg_problem(v, xsize, ysize, thick);
  if(why != NULL) {
    printf("ERROR: job %lld: %s line %lld: %s \n", job_id, jobs_path, jobs_line, why);
    exit(1);
  }
}

/* Next job header, or 0 at the end of the file; exits on a malformed header */

static int read_job(FILE *fp, char *buf, long long *job_id, int *nseg)
{
  const char *s = next_line(fp, buf);
  if(s == NULL) return 0;
  if(!parse_job(s, job_id, nseg)) {
    printf("ERROR: %s line %lld is not 'J <job_id> <n_seg>' with job_id, n_seg >= 0 \n", jobs_path, jobs_line);
    exit(1);
  }
  return 1;
}

/* Check the whole jobs file before any transport. Returns the number of jobs. */

static long long check_jobs(FILE *fp, float xsize, float ysize, float thick, long long *nseg_all)
{
  static char buf[LINESIZE];
  long long job_id, jline, njob = 0;
  int nseg, iseg;
  double v[7], nsure;
  *nseg_all = 0;
  jobs_line = 0;
  while(read_job(fp, buf, &job_id, &nseg)) {
    jline = jobs_line;
    nsure = 0.;
    for(iseg = 0; iseg < nseg; ++iseg) {
      read_seg(fp, buf, job_id, jline, v, xsize, ysize, thick);
      nsure += floor(v[6]/EPAIR);
    }
    if(nsure >= NEHSTORE) {
      printf("ERROR: job %lld (%s line %lld) makes at least %.0f e-h pairs, NEHSTORE is %d: split it \n",
             job_id, jobs_path, jline, nsure, NEHSTORE);
      exit(1);
    }
    ++njob;
    *nseg_all += nseg;
  }
  return njob;
}

/* splitmix64 finalizer: spreads the run seed over the RANLUX seed range */

static unsigned long long mix64(unsigned long long z)
{
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/* RANLUX seed of one job, in 1 .. RLUXSEEDMAX. Jobs whose ids differ by less than RLUXSEEDMAX
   get different seeds (1000003 is prime to RLUXSEEDMAX); neighbouring ids get seeds far apart. */

static int job_seed(long long seed, long long job_id)
{
  unsigned long long k;
  k = (mix64((unsigned long long)seed) % RLUXSEEDMAX + (unsigned long long)job_id % RLUXSEEDMAX) % RLUXSEEDMAX;
  return (int)(1ULL + (k * 1000003ULL) % RLUXSEEDMAX);
}

/* Empty triplg()'s buffer. triplg() keeps 120 Gaussian deviates from one RANLUX call in a static
   buffer and hands out 3 per call, so a job would start with deviates left over from the job
   before it. A refill shows as a change of the RANLUX state; 39 more calls then use up the
   buffer, and the next call (the job's first, in propag) refills it from the job's own stream. */

static void drain_triplg(void)
{
  int before[25], after[25], n;
  vect_or_f g;
  for(n = 0; n < 40; ++n) {
    rluxut_(before);
    triplg(&g);
    rluxut_(after);
    if(memcmp(before, after, sizeof(before)) != 0) break;
  }
  if(n == 40) {
    printf("ERROR: triplg() did not refill its buffer within 40 calls \n");
    exit(1);
  }
  for(n = 0; n < 39; ++n) triplg(&g);
}

/* Re-seed RANLUX (luxury level 3, as the list drivers) for one job. The first 24 numbers after
   rluxgo_() come straight from its congruential seeding, before any luxury skipping: drop them. */

static void reseed(int ins)
{
  int lux = 3, k1 = 0, k2 = 0, n24 = 24;
  float burn[24];
  fflush(stdout);
  if(dup2(null_fd, STDOUT_FILENO) < 0) {printf("ERROR: cannot redirect stdout \n"); exit(1);}
  rluxgo_(&lux, &ins, &k1, &k2);
  fflush(stdout);
  if(dup2(stdout_fd, STDOUT_FILENO) < 0) {fprintf(stderr, "ERROR: cannot restore stdout \n"); exit(1);}
  ranlux_(burn, &n24);
}

/* Main program */ int main(int argc, char *argv[])
{
    static float thick, xsize, ysize, temp, flux[2], rhe, rhh, peaktim, samptim, stimstp, pimom;
    static int ehole, new_drde, filebase;
    static int indeh[2][NEHSTORE];
    static float pixel[TXSIZE][TYSIZE];
    static float pixhist[NCRRC+1][TXSIZE][TYSIZE];
    static float crrcresp[NCRRC];
    static vect_or_f xeh[2][NEHSTORE];
    static vect_or_f xhisteh[NCRRC][2][NEHSTORE];
    static char buf[LINESIZE];
    static float rbuf[128];
    static float wthick, wxsize, wysize;
    static int wnx, wny, wnz;
    long long seed = 1, njob, nseg_all, job_id, jline, ijob, npair, ip, nout, nout_all = 0, neh_all = 0;
    int nseg, iseg, neh, all_slices = 0, i, j, k, nr, ir;
    double v[7], n, t, x, y, z, etot, zlo, zhi;
    float fx, fy, rv;
    char *end;
    struct rlimit rl;
    struct timespec t0;
    FILE *jfp, *ofp, *iifp;

    if(argc < 3 || argc > 5) {
      printf("usage: %s <jobs_file> <out_file> [seed] [last|all] \n", argv[0]);
      return 1;
    }
    jobs_path = argv[1];
    if(argc >= 4) {
      seed = strtoll(argv[3], &end, 10);
      if(*argv[3] == '\0' || *end != '\0' || seed < 0) {
        printf("ERROR: seed '%s' is not a non-negative integer \n", argv[3]);
        return 1;
      }
    }
    if(argc == 5) {
      if(strcmp(argv[4], "all") == 0) {all_slices = 1;}
      else if(strcmp(argv[4], "last") != 0) {
        printf("ERROR: slices '%s' is neither 'last' nor 'all' \n", argv[4]);
        return 1;
      }
    }
    if(getrlimit(RLIMIT_STACK, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < STACKNEED) {
      printf("ERROR: stack limit %lu kB is below the %lu kB propag() needs: ulimit -s unlimited (macOS: ulimit -s 65520) \n",
             (unsigned long)(rl.rlim_cur/1024), STACKNEED/1024);
      return 1;
    }

    /*  Define the detector parameters from the global initialization file */

    pixinit(&pimom, &thick, &xsize, &ysize, &temp, flux, &rhe, &rhh, &peaktim, &samptim, &ehole, &new_drde, &filebase);

    /* pixinit() returns 0 also when ppixel2.init is missing, when its grid exceeds NARRAYX/Y/Z
       ("Efield array too large") and when wgt_pot.init disagrees with it ("match problem"),
       leaving the field or weighting arrays unread (zero); an oversized grid would also index
       past the arrays. Transport would then run on zero fields, so stop here instead. */
    if(npixx < 2 || npixy < 2 || npixz < 2 || npixx > NARRAYX || npixy > NARRAYY || npixz > NARRAYZ) {
      printf("ERROR: field grid %d x %d x %d is missing or does not fit the %d x %d x %d arrays \n",
             npixx, npixy, npixz, NARRAYX, NARRAYY, NARRAYZ);
      return 1;
    }
    iifp = fopen("wgt_pot.init", "r");
    if(iifp == NULL || fscanf(iifp, "%f %f %f %d %d %d", &wthick, &wxsize, &wysize, &wnx, &wny, &wnz) != 6
       || wthick != thick || wxsize != xsize || wysize != ysize || wnx != npixx || wny != npixy || wnz != npixz) {
      printf("ERROR: wgt_pot.init is missing or its header differs from ppixel2.init (%f %f %f %d %d %d) \n",
             thick, xsize, ysize, npixx, npixy, npixz);
      return 1;
    }
    fclose(iifp);

    crrc(peaktim, samptim, &stimstp, crrcresp);
    printf("sample time %f \n", stimstp);

    /* Check every job before transporting any */

    jfp = fopen(jobs_path, "r");
    if(jfp == NULL) {
      printf("ERROR: cannot open jobs file %s \n", jobs_path);
      return 1;
    }
    njob = check_jobs(jfp, xsize, ysize, thick, &nseg_all);
    printf("%s: %lld jobs, %lld segments; seed %lld, %s \n", jobs_path, njob, nseg_all, seed,
           all_slices ? "all time slices" : "last time slice");
    rewind(jfp);
    jobs_line = 0;

    stdout_fd = dup(STDOUT_FILENO);
    null_fd = open("/dev/null", O_WRONLY);
    if(stdout_fd < 0 || null_fd < 0) {
      printf("ERROR: cannot open /dev/null to silence rluxgo_() \n");
      return 1;
    }
    reseed(job_seed(seed, 0));      /* initialise RANLUX before the first drain_triplg() */

    ofp = fopen(argv[2], "w");
    if(ofp == NULL) {
      printf("ERROR: cannot open output file %s \n", argv[2]);
      return 1;
    }
    fprintf(ofp,"%s \n", &header[0]);
    fprintf(ofp,"%f  %f  %f  %f\n", xsize, ysize, thick, stimstp);

    zlo = ZMARGIN;
    zhi = thick - ZMARGIN;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for(ijob = 0; read_job(jfp, buf, &job_id, &nseg); ++ijob) {
      jline = jobs_line;

      /* Fresh random stream for this job */

      drain_triplg();
      reseed(job_seed(seed, job_id));

      /* Make the e-h pairs, segment by segment */

      neh = 0;
      nout = 0;
      etot = 0.;
      for(iseg = 0; iseg < nseg; ++iseg) {
        read_seg(jfp, buf, job_id, jline, v, xsize, ysize, thick);
        etot += v[6];
        n = v[6]/EPAIR;
        npair = (long long)floor(n);
        ranlux_(&rv, &c__1);
        if(rv < n - floor(n)) {++npair;}
        if(neh + nout + npair >= NEHSTORE) {
          printf("ERROR: job %lld (%s line %lld) reaches %lld e-h pairs by segment %d, NEHSTORE is %d: split it (%s is incomplete) \n",
                 job_id, jobs_path, jline, neh + nout + npair, iseg+1, NEHSTORE, argv[2]);
          fclose(ofp);
          return 1;
        }
        for(ip = 0; ip < npair; ip += nr) {
          nr = (npair - ip < 128) ? (int)(npair - ip) : 128;
          ranlux_(rbuf, &nr);
          for(ir = 0; ir < nr; ++ir) {
            t = rbuf[ir];
            x = v[0] + t*(v[3] - v[0]);
            y = v[1] + t*(v[4] - v[1]);
            z = v[2] + t*(v[5] - v[2]);
            if(z < zlo) {z = zlo;}
            if(z > zhi) {z = zhi;}
            fx = (float)(x/xsize) + 0.5f*TXSIZE;
            fy = (float)(y/ysize) + 0.5f*TYSIZE;
            if(fx < 0.f || fx >= (float)TXSIZE || fy < 0.f || fy >= (float)TYSIZE) {
              ++nout;
              continue;
            }
            xeh[0][neh].f[0] = (float)x;
            xeh[0][neh].f[1] = (float)y;
            xeh[0][neh].f[2] = (float)z;
            xeh[0][neh].f[3] = 0.f;
            xeh[1][neh].v = xeh[0][neh].v;
            ++neh;
          }
        }
      }

      /*  Propagate the e's and h's (no signal after the sample time) */

      propag(thick, xsize, ysize, temp, flux, rhe, rhh, samptim, stimstp, ehole, neh, xeh, indeh, xhisteh);

      /*  Count e's and h's on various pixels */

      detect(xsize, ysize, thick, ehole, xhisteh, neh, crrcresp, pixel, pixhist);

      /*  Write out the results */

      fprintf(ofp,"<cluster>\n");
      fprintf(ofp,"%lld %d %d %.3f %lld\n", job_id, neh, nseg, etot, nout);
      for(k = all_slices ? 1 : NCRRC; k <= NCRRC; ++k) {
        fprintf(ofp,"<time slice %f ps>\n", k*stimstp);
        for(j = 0; j < TYSIZE; ++j) {
          for(i = 0; i < TXSIZE; ++i) {
            fprintf(ofp, i ? " %.1f" : "%.1f", pixhist[k][i][j]);
          }
          fputc('\n', ofp);
        }
      }
      neh_all += neh;
      nout_all += nout;
      if((ijob+1) % 1000 == 0) {
        printf("job %lld of %lld: %lld pairs so far, %.1f s \n", ijob+1, njob, neh_all, seconds_since(&t0));
        fflush(stdout);
      }
    }
    fclose(jfp);
    k = ferror(ofp);
    if(fclose(ofp) != 0 || k) {
      printf("ERROR: writing %s failed \n", argv[2]);
      return 1;
    }
    printf("DONE: %lld jobs, %lld pairs transported, %lld pairs outside the window, %.2f s \n",
           ijob, neh_all, nout_all, seconds_since(&t0));
    return 0;
} /* MAIN__ */

#include "ppixelav2.c"
