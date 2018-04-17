/*

   DESCRIPTION:
   Compute the weighted mean by evaluating a Gaussian to compute the weights.
       Particles are stored inside a grid so instead of searching through the entire box one limits the
       search to the particles inside the given grid and its 26 surrounding 
       neighbouring grids. For an indexed grid of 64^3 this means a speed factor of 
       64^3/27 = 9000. In reality this is slighlty less because memory access 
       is not optimized. Also it may use more memory than necesary.

    Reads Gadget files as input, it can deal with snapshots distributed
       in several files. Change variable 'files' in 'load_snapshot()'.

   MEMORY USAGE:
    Total memory needed +- extra stuff:
    3 x imagesize^3 
    5 x NPart

   USAGE:
    The only tricky part is the "imagesize[Indexed array]" option. 
    In order to speed-up the computation I generate a grid and put the
    particles there. The grid size should be as large as possible
    (not restricted to powers of two). If you use the top-hat window
    you should use a grid such that the size of each voxel is at least the
    same size as the tophat window.

   EXAMPLE:
    We want to average densities with a top-hat window of 3 Mpc. The
    input file comes from a 150 Mpc simulation (snap150Mpc). We can divide the box in
    a grid of 50x50x50 voxels.

      ./gauss_weighted_mean snap150Mpc -g 50 -w 3000 snap150Mpc.den
    
    where 50 is the grid-size and 3000 the tophat window

     __________________________________________
    |                                          |
    |  Written by Miguel Angel Aragon Calvo.   |
    |  miguel@pha.jhu.edu                      |
    |__________________________________________|


    Compile as:
       gcc  -O3 gauss_weighted_mean_1.1.c -o gauss_weighted_mean -lm -lpthread -D_REENTRANT

       gcc gauss_weighted_mean_2.0.c -o gauss_weighted_mean -O3 -D_REENTRANT -lm -fopenmp


    History:
       -Created:  21/Oct/2006 single thread code
       -Modifications:
           - 23/Oct/2006: Progress bar included.
           - 24/Oct/2006: Check segmentation fault, possibly caused by bad use of '*shell_ind'.
	   - 24/Oct/2006: Compile without warnings.
	   - 25/Oct/2006: Serious bug fixed in the indexed array
	   - 20/Apr/2009: Serious bug in passing structure to threads. 
                          The range in particles was the same for all threads!
	   - 27/Oct/2009: Add option to output floating-point densities
	   - 20/Nov/2009: Fix a bug in the range of particles sent to each thread
	   - 30/Apr/2012: Fix option for Gaussian and tophat
	   - 05/Nov/2012: Remove unnecesary pointers *density and *weight
	   - 03/Mar/2014: Fixed old segmentation fault from using density[] instead of header_info.DENSITY[] when writting to file...

	   VERSION 2.0
	   - 17/Feb/2018: Changed parallel processing to openmp
	   - 18/Feb/2018: Add preprocessor directives to avoid conditionals in code. Removed "tophat option"
	                  Add recursive density computation


*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <time.h>


#define FLAG_GAUSS
//#define FLAG_SAMPLING

//===============================================================
//--- Global variables
//===============================================================
long   *ngp_n, *ngp_o, *ngp_i, *IDs;
//--- Temporal variables
double TempD;
float  TempF;


struct particle_data {
  double  Pos[3];
} *P, *P2;     //--- Pointer to structure


//--- Gadget header structure
struct io_header_1
{
  int      npart[6];
  double   mass[6];
  double   time;
  double   redshift;
  int      flag_sfr;
  int      flag_feedback;
  int      npartTotal[6];
  int      flag_cooling;
  int      num_files;
  double   BoxSize;
  double   Omega0;
  double   OmegaLambda;
  double   HubbleParam; 
  char     fill[256- 6*4- 6*8- 2*8- 2*4- 6*4- 2*4 - 4*8];  //--- Fills to 256 Bytes 
} header1;


struct header_struct {
  //---  General info
  int    N_Data;             //--- Number  of data     points
  int    N_Sampling;         //--- Number  of sampling points
  double Boxsize_Data;       //--- Boxsize of data     points
  double Boxsize_Samplings;  //--- Boxsize of sampling points
  int    GridSize;           //--- Grid used for indexed array
  double WindowSize;         //--- Size of window used to compute density
  //---  Flags
  char   Sampling;           //--- Use sampling points instead of data points
  char   Weights;            //--- Read weights from file
  char   Gadget;             //--- Do gadget file
  char   Float_den;          //--- Write floating point densities
  //---  Files
  char   file_input   [256];
  char   file_output  [256];
  char   file_sampling[256];
  char   file_weights [256];
  //---  Needed for threads
  int    N_Threads;
} header_info;     //--- Pointer to structure


double *DENSITY, *WEIGHT, *GAUSS_LOOKUP;

//===============================================================
//================   FUNCTIONS DECLARATION  =====================
//===============================================================

struct particle_data *read_particles(char *fname, int *N_part, double *Boxsize);
struct particle_data *read_particles_gadget(char *fname, int files, int *N_part, double *BoxSize);
void    create_indexed_array(int GridSize, double BoxSize, int N_part);
void    get_densities();
void    save_density(char *fname, int N_part, double *density);
long   *lvector(long nh);
float  *vector(long nh);
double *dvector(long nh);
long   *shell(long *shell_ind, long nx, long ny, long nz, long n_grid);
void    read_command_line(int argc, char **argv, struct header_struct *header_info);
double *read_weight(char *fname);


//==============================================
//================   MAIN  =====================
//==============================================
int main(int argc, char **argv)
{

  int       i,j,k;
  int       i_med;                //--- Helper variable to divide work in thread
  
  //--- Timer
  time_t t1,t2;
  (void) time(&t1);


  //------------------------------
  //--- Read command line options
  //------------------------------
  read_command_line(argc, argv, &header_info);

  //------------------------------
  //--- Read particles from file
  //------------------------------
  if (header_info.Gadget == 0)
    P  = read_particles(       header_info.file_input, &header_info.N_Data, &header_info.Boxsize_Data);  
  else
    P  = read_particles_gadget(header_info.file_input, header_info.Gadget, &header_info.N_Data, &header_info.Boxsize_Data);  

  //--- By default each data point is a sampling point
  header_info.N_Sampling = header_info.N_Data; 
  
  //------------------------------
  //--- Read data points from file
  //------------------------------
  #ifdef FLAG_SAMPLING
    P2 = read_particles(header_info.file_sampling, &header_info.N_Sampling, &header_info.Boxsize_Samplings);
  #endif

  //------------------------------
  //--- Allocate memory for density
  //------------------------------
  #ifdef FLAG_SAMPLING
    printf("Allocating %d for sampling density\n",header_info.N_Sampling);
    DENSITY = dvector(header_info.N_Sampling);
  #else
    printf("Allocating %d for data density\n",header_info.N_Data);
    DENSITY = dvector(header_info.N_Data);
  #endif

  //------------------------------
  //--- Read weights from file
  //------------------------------
  if(header_info.Weights == 1) WEIGHT = read_weight(header_info.file_weights);

  //------------------------------
  //--- Create indexed array
  //------------------------------
  create_indexed_array(header_info.GridSize, header_info.Boxsize_Data, header_info.N_Data);
  
  //------------------------------
  //   Compute densities
  //------------------------------
  get_densities();
  get_densities();
  get_densities();
  get_densities();


  printf("\nSaving densities...\n");
  //------------------------------
  //--- Save densities to file
  //------------------------------
  save_density(header_info.file_output, header_info.N_Sampling, DENSITY);
  
  //--- Check time and print
  (void) time(&t2);
  printf("\nComputing time = %d seconds\n", (int) t2-t1);

  free(DENSITY);
  free(WEIGHT);
  free(P);

  #ifdef FLAG_SAMPLING
    free(P2);
  #endif

  return(0);

}


//==================================================================
//           FUNCTIONS
//==================================================================

//========================================
//--- 
//========================================
void read_command_line(int argc, char **argv, struct header_struct *header_info){

  int i;

  //-------------------------------
  //--- Default options
  //-------------------------------
  header_info->N_Threads = 1;
  header_info->Gadget    = 0;
  header_info->Float_den = 0;
  
  //-------------------------------
  //--- Read command line options
  //-------------------------------
  printf("\n-------------------------------------------------\n");
  for (i = 1; i < argc; i++) { 
    //--- OUTPUT FILE
    if      (strcmp( "-o", argv[i]) == 0){
      sscanf(argv[i+1],"%s",&header_info->file_output);
      printf(">>>     Output file = %s \n", header_info->file_output);
    }
    //--- INPUT POSITIONS
    else if (strcmp( "-i", argv[i]) == 0){
      sscanf(argv[i+1],"%s",&header_info->file_input);
      printf(">>>     Input file = %s \n", header_info->file_input);
    }
    //--- FLOATING POINT OUTPUT
    else if (strcmp( "-f", argv[i]) == 0){
      header_info->Float_den = 1;
      printf(">>>     Output floating-point values \n");
    }
    //--- READ WINDOW SIZE
    else if (strcmp( "-w", argv[i]) == 0){
      sscanf(argv[i+1],"%lf",&header_info->WindowSize);
      printf(">>>     Window size = %lf\n", header_info->WindowSize);
    }
    //--- READ GRID SIZE FOR INDEXING
    else if (strcmp( "-g", argv[i]) == 0){
      sscanf(argv[i+1],"%d",&header_info->GridSize);  
      printf(">>>     Grid   size = %d\n", header_info->GridSize);
    }
    //--- SAMPLING FILE
    else if (strcmp( "-s", argv[i]) == 0){
      header_info->Sampling = 1;
      sscanf(argv[i+1],"%s",&header_info->file_sampling);  
      printf(">>>     Sampling file: '%s'\n",header_info->file_sampling);
    }
    //--- WEIGHTS
    else if (strcmp( "-p", argv[i]) == 0){
      header_info->Weights = 1;
      sscanf(argv[i+1],"%s",&header_info->file_weights);
      printf(">>>     Weights file: '%s'\n",header_info->file_weights);
    }
    //--- NUMBER OF THREADS
    else if (strcmp( "-T", argv[i]) == 0){
      sscanf(argv[i+1],"%i",&header_info->N_Threads);
      printf(">>>     Running on %d threads\n",header_info->N_Threads);
    }
    //--- USE GADGET FILE
    else if (strcmp( "-G", argv[i]) == 0){
      sscanf(argv[i+1],"%i",&header_info->Gadget);
      printf(">>>     Read Gadget file in %d files\n",header_info->Gadget);
    }

  }

  //-------------------------------
  //--- Some extra info
  //-------------------------------
  if (header_info->N_Threads == 1) printf(">>>     Running default mode: 1 thread\n");


  //--- Running info
  if (argc < 2) {
    printf("Usage: \n");
    printf("       ./gauss_weighted_mean\n");
    printf("                       -i   [Input file]    \n");
    printf("                       -o   [Output file]   \n");
    printf("                       <-f> [Floating point densities (default is double)]\n");
    printf("                             NOTE: when this flag is set also the Weights are expected as float.\n");
    printf("                       -w   [Window size]   \n");
    printf("                       -g   [Grid size]     \n");
    printf("                       -s   [Sampling file] \n");
    printf("                             NOTE: Use this option ONLY when compiled with FLAG_SAMPLING option.\n");
    printf("                       -p   [Weights] \n");
    printf("                       -T   [Threads] \n");
    printf("                       -G   [Use Gadget file, followed by number of files] \n\n");
    printf("        < > indicates an optional flag with no parameters.\n\n");         
    exit(0);
  }

  printf("-------------------------------------------------\n\n");

}

//========================================
//--- Read positions from a raw file
//========================================
struct particle_data *read_particles(char *fname, int *N_part, double *BoxSize)
{
  struct particle_data *Part = NULL;             //--- Pointer to structure
  FILE *fd;
  char buf[200];
  int  i,j,k,dummy,ntot_withmasses;
  int  t,n,off,pc,pc_new,pc_sph;
  int  count_err=0;

  printf(">> Raw %s", fname);
  sprintf(buf,"%s",fname);
  printf("Attempting to open raw file %s\n...", buf);
  
  if(!(fd=fopen(buf,"r"))){
    printf("can't open raw file '%s'\n",buf);
    exit(0);}
  
  //--- Read number of particles
  fread(&TempF, sizeof(float), 1, fd);
  fread(N_part, sizeof(int), 1, fd);
  *BoxSize = (double) TempF;
  printf("Reading %d particles inside Box of side %f from file '%s' ...\n", *N_part, *BoxSize,  buf); fflush(stdout);

  //--- Allocate memory for particles
  printf("   Allocating memory for positions...\n"); fflush(stdout);
  Part=(struct particle_data *) malloc((size_t) ( (*N_part) * sizeof(struct particle_data)));
  if (!Part) {
    printf("Error alocating memory for particles"); fflush(stdout);
    exit(0);
  }

  //--- Read particles
  printf("   Reading positions...\n"); fflush(stdout);
  for(n=0; n< (*N_part); n++) {
    fread(&TempF, sizeof(float), 1, fd);
    Part[n].Pos[0] = (double) TempF;
    fread(&TempF, sizeof(float), 1, fd);
    Part[n].Pos[1] = (double) TempF;
    fread(&TempF, sizeof(float), 1, fd);
    Part[n].Pos[2] = (double) TempF;
  }

  fclose(fd);

  //--- Fix boundaries (x >= BoxSize || x < 0)
  printf("   Checking periodic conditions...\n"); fflush(stdout);
  for(i=0; i<= (*N_part)-1; i++){
    if(Part[i].Pos[0] >= *BoxSize){ Part[i].Pos[0] = Part[i].Pos[0] - (*BoxSize); count_err++;}
    if(Part[i].Pos[1] >= *BoxSize){ Part[i].Pos[1] = Part[i].Pos[1] - (*BoxSize); count_err++;}
    if(Part[i].Pos[2] >= *BoxSize){ Part[i].Pos[2] = Part[i].Pos[2] - (*BoxSize); count_err++;}
    if(Part[i].Pos[0] <       0.0){ Part[i].Pos[0] = Part[i].Pos[0] + (*BoxSize); count_err++;}
    if(Part[i].Pos[1] <       0.0){ Part[i].Pos[1] = Part[i].Pos[1] + (*BoxSize); count_err++;}
    if(Part[i].Pos[2] <       0.0){ Part[i].Pos[2] = Part[i].Pos[2] + (*BoxSize); count_err++;}
  }
  printf("   %d particles out of box\n", count_err); fflush(stdout);

  return Part;
}

//========================================
//--- Read positions from a Gadget File
//========================================
struct particle_data *read_particles_gadget(char *fname, int files, int *N_part, double *BoxSize)
{
  FILE *fd;
  char buf[200];
  int  i,k, dummy, ntot_withmasses;
  int  n,pc,pc_new;
  int  count_err;

  struct particle_data *Part = NULL;             //--- Pointer to structure

  count_err=0;

  sprintf(buf,"%s", fname);
  printf("Attempting to open Gadget file %s in [%d] files...\n", buf, files);


#define SKIP fread(&dummy, sizeof(dummy), 1, fd);
  
  for(i=0, pc=1; i<files; i++, pc=pc_new){

      if(files>1)
	sprintf(buf,"%s.%d",fname,i);
      else
	sprintf(buf,"%s",fname);
      
 
      if(!(fd=fopen(buf,"r"))){
	printf("can't open Gadget file '%s'\n",buf);
	exit(0);
      }
      
      printf("Reading Gadget file '%s'...\n",buf); fflush(stdout);
      
      fread(&dummy, sizeof(dummy), 1, fd);     
      fread(&header1, sizeof(header1), 1, fd);  //--- Read Header1 in one pass
      fread(&dummy, sizeof(dummy), 1, fd);

      //--- Boxsize to global variable
      *BoxSize = (double) header1.BoxSize;
      
      //--- Number of particles
      for(k=0, *N_part=0, ntot_withmasses=0; k<5; k++)
	*N_part+= header1.npart[k];
      
      //--- Allocate memory for particles. Do the first loop
      if (i == 0) {
	printf("   Allocating memory for positions...\n"); fflush(stdout);
	Part=(struct particle_data *) malloc((size_t) ( (*N_part) * sizeof(struct particle_data)));
	if (!Part) {
	  printf("Error alocating memory for particles"); fflush(stdout);
	  exit(0);
	} //--- end if (!Part)
	
	printf("Boxsize = %f", *BoxSize);

      } //--- end if (i == 0)

      //--- Particles with mass. Use weights here
      for(k=0, ntot_withmasses=0; k<5; k++){
	if(header1.mass[k]==0)
	    ntot_withmasses+= header1.npart[k];
      }
            
      SKIP;
      
      //--- Read Particle's positions. 

      for(n=0; n< (*N_part); n++) {
	fread(&TempF, sizeof(float), 1, fd);
	Part[n].Pos[0] = (double) TempF;
	fread(&TempF, sizeof(float), 1, fd);
	Part[n].Pos[1] = (double) TempF;
	fread(&TempF, sizeof(float), 1, fd);
	Part[n].Pos[2] = (double) TempF;
      }

      fclose(fd);
  }

  //--- Fix boundaries (x >= BoxSize || x < 0)
  printf("   Checking periodic conditions...\n");
  for(i=0; i<*N_part; i++){
    if(Part[i].Pos[0] >= *BoxSize){ Part[i].Pos[0] = Part[i].Pos[0] - *BoxSize; count_err++;}
    if(Part[i].Pos[1] >= *BoxSize){ Part[i].Pos[1] = Part[i].Pos[1] - *BoxSize; count_err++;}
    if(Part[i].Pos[2] >= *BoxSize){ Part[i].Pos[2] = Part[i].Pos[2] - *BoxSize; count_err++;}
    if(Part[i].Pos[0] <       0.0){ Part[i].Pos[0] = Part[i].Pos[0] + *BoxSize; count_err++;}
    if(Part[i].Pos[1] <       0.0){ Part[i].Pos[1] = Part[i].Pos[1] + *BoxSize; count_err++;}
    if(Part[i].Pos[2] <       0.0){ Part[i].Pos[2] = Part[i].Pos[2] + *BoxSize; count_err++;}
  }
  printf("   %d particles out of box\n", count_err);
  

  return Part;
}



//=====================================
//--- Read weight file
//=====================================
double *read_weight(char *fname)
{
  FILE   *fd;
  char   buf[200];
  int    i;
  double TempD;
  float  TempF;
  int    N_part;
  double *weight;

  //--- Load filename
  sprintf(buf,"%s",fname);
  printf("Reading weights from file '%s' ...\n",buf); fflush(stdout);
  if(!(fd=fopen(buf,"r"))){
    printf("can't open file '%s'\n",buf);
    exit(0);}

  //--- Header of file (number of partices)
  fread(&N_part,sizeof(int),1,fd);

  //--- Allocate memory. OJO: for now all computaions are done in double...
  weight = (double *) malloc((size_t) ( N_part * sizeof(double)));
  if (!weight) {
    printf("Error alocating memory for particles"); fflush(stdout);
    exit(0);
  }

  //--- Read weights...
  printf("Reading %d weights \n", N_part);
  if (header_info.Float_den == 0){
    for (i=0; i<N_part; i++) {
      fread(&TempD,sizeof(double),1,fd);
      weight[i] = TempD;
    }
  } else {
    for (i=0; i<N_part; i++) {
      fread(&TempF,sizeof(float),1,fd);
      weight[i] = (double) TempF;
    }
  }

  fclose(fd);
  printf("     Ready reading weights.\n");

  return weight;

}


//========================================
//--- Main function to compute densities
//========================================
void get_densities(){


#pragma omp parallel
  {  

  long   i,j,k,w;
  int    cont=0;
  long   xgrid,ygrid,zgrid;
  long   Ind_Shell[27];                     //--- Array of long
  long   *shell_ind;                        //--- Pointer to long
  long   neighbour_id;
  double xi,yi,zi,xp,yp,zp;
  double denomin_cumulative;   
  double density_cumulative;
  double dista2;
  double weight_temp = 1.0;                 //--- Default unitary weight
  double gauss_eval;
  
  //--- Handy instead of using the complete structure
  double BoxSize    = header_info.Boxsize_Data;
  double sigma      = header_info.WindowSize;
  int    GridSize   = header_info.GridSize;
  double Weights    = header_info.Weights;

  //--- Initialize variables...
  shell_ind = Ind_Shell;
  double BoxSize2         = BoxSize/2.0;
  double sigma2           = sigma*sigma;
  double sigma_3          = sigma*3.0;
  double sigma_4          = sigma*4.0;
  double GridSize_BoxSize = GridSize/BoxSize;
  
  //-----------------------------
  //--- Preamble to main loop
  //-----------------------------

  #pragma omp for

  //--- Main density loop. Use different ranges for each thread
  for(i=0; i<header_info.N_Data; i++){
    
    if ( i % 10000 == 0 ) printf("%d, ", i);
    
    density_cumulative = 0.0;
    denomin_cumulative = 0.0;

    //--- Particle i
    #ifdef FLAG_SAMPLING
      xi = P2[i].Pos[0];
      yi = P2[i].Pos[1];
      zi = P2[i].Pos[2];
    #else
      xi = P[i].Pos[0];
      yi = P[i].Pos[1];
      zi = P[i].Pos[2];
    #endif

    //--- Convert to grid coordinates
    xgrid = xi * GridSize_BoxSize;
    ygrid = yi * GridSize_BoxSize;
    zgrid = zi * GridSize_BoxSize;

    //--- Get shell around [xgrid,ygrid,zgrid] cell
    shell_ind = shell(shell_ind, xgrid,ygrid,zgrid, GridSize);

    //--- Loop over shell...
    for(k=0; k<27; k++){
      
      //--- Skip empty cells
      if(ngp_n[shell_ind[k]] == 0) continue;

      //--- Loop over particles inside cell
      for(j=0; j<ngp_n[shell_ind[k]]; j++){

	//--- ID of neighbour particle
	neighbour_id = ngp_i[ngp_o[shell_ind[k]]+j];
	
	//--- Distance to neighbour particle
	xp = xi - P[neighbour_id].Pos[0];
	yp = yi - P[neighbour_id].Pos[1];
	zp = zi - P[neighbour_id].Pos[2];

	//--- Periodic boundaries...
	if(xp < -BoxSize2) xp = xp + BoxSize;
	if(xp >  BoxSize2) xp = xp - BoxSize;
	if(yp < -BoxSize2) yp = yp + BoxSize;
	if(yp >  BoxSize2) yp = yp - BoxSize;
	if(zp < -BoxSize2) zp = zp + BoxSize;
	if(zp >  BoxSize2) zp = zp - BoxSize; 	


	//----------------------------------
	//--- Gaussian densities
	//----------------------------------
	#ifdef FLAG_GAUSS
	  //--- Skip distant particles...
	  if((fabs(xp) > sigma_3) || (fabs(yp) > sigma_3) ||(fabs(zp) > sigma_3) ) continue;
	  
	  //--- Squared distance from particle
	  dista2 = xp*xp + yp*yp + zp*zp;

	  //--- Gaussian
	  gauss_eval = exp( -(dista2/2.0) / sigma2 );
	  density_cumulative = density_cumulative + gauss_eval * WEIGHT[neighbour_id];
	  denomin_cumulative = denomin_cumulative + 1;

	//----------------------------------
	//--- TopHat densities
	//----------------------------------
	#else
	
	  //--- Squared distance from particle
	  dista2 = xp*xp + yp*yp + zp*zp;

	  //--- Tophat
	  if(dista2 < sigma2){
	    density_cumulative = density_cumulative + WEIGHT[neighbour_id];
	    denomin_cumulative = denomin_cumulative + 1;
	  } // --- if(dista2 < sigma2)
	  
	#endif

      } //--- for(j=0; j<ngp_n[shell_ind[k]]; j++)

    } //--- for(k=0; k<27; k++)

    //--- Store density in array
    DENSITY[i] = density_cumulative / denomin_cumulative;
    
    cont++;

  } //--- for(i=ini; i<fin; i++)
  printf("\n");

  //--- Make sure all threads are ready before continuing
#pragma omp barrier

  } // #pragma omp parallel for private(i)
  
}


//=====================================
//--- Creates Gaussian lookup
//=====================================
//double create_indexed_array(int GridSize, double BoxSize, int N_part)


//=====================================
//--- Creates the indexed array
//=====================================
void create_indexed_array(int GridSize, double BoxSize, int N_part)
{
  int  i;
  int  xgrid,ygrid,zgrid;
  int  ind;  //--- Linear index
  int  N_GridSize;

  N_GridSize = GridSize*GridSize*GridSize;

  printf("Prepare to allocate %ld cells\n", N_GridSize);
  //--- Create Ids array
  IDs     = lvector(N_part);
  //--- Create 1D image
  ngp_n   = lvector(N_GridSize);
  //--- Create 1D offset
  ngp_o   = lvector(N_GridSize);
  //--- Create 1D indexed array
  ngp_i   = lvector(N_part);


  //--- IDs
  printf("Create Id's array... "); fflush(stdout);
  for(i=0; i<N_part; i++) IDs[i] = i;
  printf("done\n"); fflush(stdout);

  //--- Image
  for(i=0; i<N_GridSize; i++) ngp_n[i] = 0;
  printf("Creating image array... "); fflush(stdout);
  for(i=0; i<N_part; i++){
    xgrid = (int) ((P[i].Pos[0]/BoxSize)*GridSize);
    ygrid = (int) ((P[i].Pos[1]/BoxSize)*GridSize);
    zgrid = (int) ((P[i].Pos[2]/BoxSize)*GridSize);
    //--- Linear index
    ind   = xgrid + ygrid*GridSize + zgrid*GridSize*GridSize;
    //--- Increase number of galaxies per grid
    ngp_n[ind] = ngp_n[ind] + 1; 
  }
  printf("done\n"); fflush(stdout);

  //--- Offset
  for(i=0; i<N_GridSize; i++) ngp_o[i] = 0;
  printf("Creating offset array... "); fflush(stdout);
  for(i=1; i<N_GridSize; i++) ngp_o[i] = ngp_o[i-1] + ngp_n[i-1];
  printf("done\n"); fflush(stdout);

  //--- Index
  printf("Creating indexed array... "); fflush(stdout);
  for(i=0; i<N_part; i++){
    xgrid = (long) ((P[i].Pos[0]/BoxSize)*GridSize);
    ygrid = (long) ((P[i].Pos[1]/BoxSize)*GridSize);
    zgrid = (long) ((P[i].Pos[2]/BoxSize)*GridSize);
    //--- Linear index
    ind   = xgrid + ygrid*GridSize + zgrid*GridSize*GridSize;

    //--- Avoid empty cells
    if(ngp_n[ind] == 0) continue;

    //--- Fill indexed array with Id's
    ngp_i[ngp_o[ind]] = IDs[i];  
    //--- Increase the offset in cell to let room for new particle,
    //    Note that this is destructive operation, we must compute
    //    ngp_o again...
    ngp_o[ind] = ngp_o[ind] + 1; 
  }
  printf("done\n"); fflush(stdout);


  for(i=0; i<N_GridSize; i++) ngp_o[i] = 0;
  printf("Creating offset array again... "); fflush(stdout);
  for(i=1; i<N_GridSize; i++) ngp_o[i] = ngp_o[i-1] + ngp_n[i-1];
  printf("done\n"); fflush(stdout);


}


//=====================================
//--- Writes density file
//=====================================
void save_density(char *fname, int N_part, double *density)
{
  FILE *fd;
  char buf[200];
  int  i;
  double tempd; 
  float  tempf;

  //--- Load filename
  sprintf(buf,"%s",fname);
  
  printf("Writting densities to file '%s' ...\n",buf); fflush(stdout);
  fd=fopen(buf,"wb");
  //--- Header of file (number of partices)
  fwrite(&N_part,sizeof(int),1,fd);

  //--- Write densities to file
  if (header_info.Float_den == 0){
    printf("Writting %d [double] densities \n", N_part);
    for (i=0; i<N_part; i++) {
      fwrite(&DENSITY[i],sizeof(double),1,fd);
    }
  }
  else {
    printf("Writting %d [float] densities \n", N_part);
    for (i=0; i<N_part; i++) {
      tempf = (float) DENSITY[i];
      fwrite(&tempf,sizeof(float),1,fd);
    }
  }

  fclose(fd);
  printf("     Ready writting density file.\n");
  
}


//=====================================
//--- Numerical Recipes standard error handler
//=====================================
void nrerror(char error_text[])
{
  fprintf(stderr,"Numerical Recipes run-time error...\n");
  fprintf(stderr,"%s\n",error_text);
  fprintf(stderr,"...now exiting to system...\n");
  exit(1);
}

//=====================================
//--- Allocate an integer vector with subscript range v[nl..nh]
//=====================================
long *lvector(long nh)
{
  long *v;

  v=(long *)malloc((size_t) ((nh)*sizeof(long)));
  if (!v) nrerror("allocation failure in lvector()");
  printf("Allocated %ld cells.\n", nh);
  return v;
}

//=====================================
//--- allocate a float vector with subscript range v[nl..nh]
//=====================================
double *dvector(long nh)
{
  double *v;
  v=(double *)malloc((size_t) ((nh)*sizeof(double)));
  if (!v) nrerror("allocation failure in double vector()");
  return v;
}

//=====================================
//--- Get indexes of shell around cell
//=====================================
long *shell(long *shell_ind, long nx, long ny, long nz, long n_grid)
{
  long x00,x01,x02,x03,x04,x05,x06,x07,x08,x09;
  long x10,x11,x12,x13,x14,x15,x16,x17,x18,x19;
  long x20,x21,x22,x23,x24,x25,x26;
  long y00,y01,y02,y03,y04,y05,y06,y07,y08,y09;
  long y10,y11,y12,y13,y14,y15,y16,y17,y18,y19;
  long y20,y21,y22,y23,y24,y25,y26;
  long z00,z01,z02,z03,z04,z05,z06,z07,z08,z09;
  long z10,z11,z12,z13,z14,z15,z16,z17,z18,z19;
  long z20,z21,z22,z23,z24,z25,z26;
  
  //--- X
  x00 = nx;
  x01 = nx-1;
  x02 = nx;
  x03 = nx+1;
  x04 = nx-1;
  x05 = nx;
  x06 = nx+1;
  x07 = nx-1;
  x08 = nx;
  x09 = nx+1;
  x10 = nx-1;
  x11 = nx;
  x12 = nx+1;
  x13 = nx-1;
  x14 = nx+1;
  x15 = nx-1;
  x16 = nx;
  x17 = nx+1;
  x18 = nx-1;
  x19 = nx;
  x20 = nx+1;
  x21 = nx-1;
  x22 = nx;
  x23 = nx+1;
  x24 = nx-1;
  x25 = nx;
  x26 = nx+1;
  //--- Y
  y00 = ny;
  y01 = ny-1;
  y02 = ny-1;
  y03 = ny-1;
  y04 = ny;
  y05 = ny;
  y06 = ny;
  y07 = ny+1;
  y08 = ny+1;
  y09 = ny+1;
  y10 = ny-1;
  y11 = ny-1;
  y12 = ny-1;
  y13 = ny;
  y14 = ny;
  y15 = ny+1;
  y16 = ny+1;
  y17 = ny+1;
  y18 = ny-1;
  y19 = ny-1;
  y20 = ny-1;
  y21 = ny;
  y22 = ny;
  y23 = ny;
  y24 = ny+1;
  y25 = ny+1;
  y26 = ny+1;
  //--- Z
  z00 = nz;
  z01 = nz-1;
  z02 = nz-1;
  z03 = nz-1;
  z04 = nz-1;
  z05 = nz-1;
  z06 = nz-1;
  z07 = nz-1;
  z08 = nz-1;
  z09 = nz-1;
  z10 = nz;
  z11 = nz;
  z12 = nz;
  z13 = nz;
  z14 = nz;
  z15 = nz;
  z16 = nz;
  z17 = nz;
  z18 = nz+1;
  z19 = nz+1;
  z20 = nz+1;
  z21 = nz+1;
  z22 = nz+1;
  z23 = nz+1;
  z24 = nz+1;
  z25 = nz+1;
  z26 = nz+1;
  
  
  //--- Periodic conditions
  if(nx == 0){
    x01 = n_grid-1;
    x04 = n_grid-1;
    x07 = n_grid-1;
    x10 = n_grid-1;
    x13 = n_grid-1;
    x15 = n_grid-1;
    x18 = n_grid-1;
    x21 = n_grid-1;
    x24 = n_grid-1;}
  
  if(nx == n_grid-1){
    x03 = 0;
    x06 = 0;
    x09 = 0;
    x12 = 0;
    x14 = 0;
    x17 = 0;
    x20 = 0;
    x23 = 0;
    x26 = 0;}
  
  if(ny == 0){
    y01 = n_grid-1;
    y02 = n_grid-1;
    y03 = n_grid-1;
    y10 = n_grid-1;
    y11 = n_grid-1;
    y12 = n_grid-1;
    y18 = n_grid-1;
    y19 = n_grid-1;
    y20 = n_grid-1;}
  
  if(ny == n_grid-1){
    y07  = 0;
    y08  = 0;
    y09  = 0;
    y15  = 0;
    y16  = 0;
    y17  = 0;
    y24  = 0;
    y25  = 0;
    y26  = 0;}

  if(nz == 0){
    z01 = n_grid-1;
    z02 = n_grid-1;
    z03 = n_grid-1;
    z04 = n_grid-1;
    z05 = n_grid-1;
    z06 = n_grid-1;
    z07 = n_grid-1;
    z08 = n_grid-1;
    z09 = n_grid-1;}

  if(nz == n_grid-1){
    z18 = 0;
    z19 = 0;
    z20 = 0;
    z21 = 0;
    z22 = 0;
    z23 = 0;
    z24 = 0;
    z25 = 0;
    z26 = 0;}

  //--- Evaluate the shell_ind
  shell_ind[0] = x00 + y00*n_grid + z00*n_grid*n_grid;
  shell_ind[1] = x01 + y01*n_grid + z01*n_grid*n_grid;
  shell_ind[2] = x02 + y02*n_grid + z02*n_grid*n_grid;
  shell_ind[3] = x03 + y03*n_grid + z03*n_grid*n_grid;
  shell_ind[4] = x04 + y04*n_grid + z04*n_grid*n_grid;
  shell_ind[5] = x05 + y05*n_grid + z05*n_grid*n_grid;
  shell_ind[6] = x06 + y06*n_grid + z06*n_grid*n_grid;
  shell_ind[7] = x07 + y07*n_grid + z07*n_grid*n_grid;
  shell_ind[8] = x08 + y08*n_grid + z08*n_grid*n_grid;
  shell_ind[9] = x09 + y09*n_grid + z09*n_grid*n_grid;
  shell_ind[10] = x10 + y10*n_grid + z10*n_grid*n_grid;
  shell_ind[11] = x11 + y11*n_grid + z11*n_grid*n_grid;
  shell_ind[12] = x12 + y12*n_grid + z12*n_grid*n_grid;
  shell_ind[13] = x13 + y13*n_grid + z13*n_grid*n_grid;
  shell_ind[14] = x14 + y14*n_grid + z14*n_grid*n_grid;
  shell_ind[15] = x15 + y15*n_grid + z15*n_grid*n_grid;
  shell_ind[16] = x16 + y16*n_grid + z16*n_grid*n_grid;
  shell_ind[17] = x17 + y17*n_grid + z17*n_grid*n_grid;
  shell_ind[18] = x18 + y18*n_grid + z18*n_grid*n_grid;
  shell_ind[19] = x19 + y19*n_grid + z19*n_grid*n_grid;
  shell_ind[20] = x20 + y20*n_grid + z20*n_grid*n_grid;
  shell_ind[21] = x21 + y21*n_grid + z21*n_grid*n_grid;
  shell_ind[22] = x22 + y22*n_grid + z22*n_grid*n_grid;
  shell_ind[23] = x23 + y23*n_grid + z23*n_grid*n_grid;
  shell_ind[24] = x24 + y24*n_grid + z24*n_grid*n_grid;
  shell_ind[25] = x25 + y25*n_grid + z25*n_grid*n_grid;
  shell_ind[26] = x26 + y26*n_grid + z26*n_grid*n_grid;
  
  return shell_ind;
}


