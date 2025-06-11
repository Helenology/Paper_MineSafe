//#include "allocation.h"
#ifndef READ_H_INCLUDED
#define READ_H_INCLUDED
#define M 240
#define N 360
#define C 3

typedef struct header{
	char head[2];
	int height;
	int width;
	int depth;
	int intensity;
}img_header;
img_header img1;
void reintialize_img_header()
{

     img1.height=0;
     img1.width=0;
     img1.depth=0;
     img1.intensity=0;

}

int ***img_read_head_ppm(char *file_name)
{

      int ***img_array,row,col,d_pth,i,j;
char tx[26];
	//tx=(char*)calloc(25,sizeof(char));
      FILE *fp1;
      fp1=fopen(file_name,"r");

// 	fscanf(fp1,"%s\n",img1.head);
// 	fscanf(fp1,"%[^\n]",tx);        
// 	fscanf(fp1,"%d\t",&img1.width);
// 	fscanf(fp1,"%d\n",&img1.height);
// 	fscanf(fp1,"%d\n",&img1.intensity);
    img1.depth=3;
	img_array=(int***)calloc(M,sizeof(int**));
	for(i=0;i<M;i++) 
		{
		img_array[i]=(int**)calloc(N,sizeof(int*));
                }
        for(i=0;i<M;i++){
		for(j=0;j<N;j++){				
			img_array[i][j]=(int*)calloc(C,sizeof(int));}}

		for(row=0;row<M;row++)
		{
			for(col=0;col<N;col++)
			{
			  for(d_pth=0;d_pth<C;d_pth++)
			  {
				fscanf(fp1,"%d",&img_array[row][col][d_pth]);
			  }
		        }
	        }
//free(tx);
	return img_array;
}

int ***img_read_ppm(char *file_name)
{
	
      int ***img_array,ro,cl,d_pth,i,j;
	img_array=(int***)calloc(M,sizeof(int**));
	for(i=0;i<M;i++) 
		{
		img_array[i]=(int**)calloc(N,sizeof(int*));
                }
        for(i=0;i<M;i++){
		for(j=0;j<N;j++){				
			img_array[i][j]=(int*)calloc(C,sizeof(int));}}
    printf("hi start reading file\n");
      FILE *fp1;
      fp1=fopen(file_name,"r");
  printf("Every thing is ok\n");
	
		for(ro=0;ro<M;ro++)
		{
			for(cl=0;cl<N;cl++)
			{
			  for(d_pth=0;d_pth<C;d_pth++)
			  {
				fscanf(fp1,"%d",&img_array[ro][cl][d_pth]);
			  }
		        }
	        }
  printf("End reading file\n");
	return img_array;
	fclose(fp1);
}
int **img_read_pgm(char *file_name)
{
      int ro,cl;
      int **img_array;
      FILE *fp1;
      fp1=fopen(file_name,"rb+");
	

		for(ro=0;ro<M;ro++)
		{
			for(cl=0;cl<N;cl++)
			{

				fscanf(fp1,"%d",&img_array[ro][cl]);
			  }

	        }
	        fclose (fp1);
	return img_array;
}
void img_write_ppm(int ***img_array,char *file_name)
{

      FILE *fp1;
      int ro,cl,depth;
      fp1=fopen(file_name,"w");
// 	fprintf(fp1,"P3\n");
// 	fprintf(fp1,"%d\t%d\n",N,M);
// 	fprintf(fp1,"255\n");
		for(ro=0;ro<M;ro++)
		{
			for(cl=0;cl<N;cl++)
			{
			  for(depth=0;depth<C;depth++)
			   {
				fprintf(fp1,"%d\n",img_array[ro][cl][depth]);
			   }

			  }

		  }

fclose(fp1);
}
void img_write_pgm(int **img_array,char *file_name)
{

      FILE *fp1;
      int ro,cl,i;
      fp1=fopen(file_name,"w");
       	fprintf(fp1,"P2\n");
	fprintf(fp1,"%d\t%d\n",N,M);
	fprintf(fp1,"255\n");
        for(ro=0;ro<M;ro++)
	   {
	    for(cl=0;cl<N;cl++)
		{
		fprintf(fp1,"%d\n",img_array[ro][cl]);
		}
	    }
	fclose(fp1);

}
void img_write_pgm_f(float **img_array,char *file_name)
{

      FILE *fp1;
      int ro,cl;
      fp1=fopen(file_name,"wb+");

		for(ro=0;ro<M;ro++)
		{
			for(cl=0;cl<N;cl++)
			{

				fprintf(fp1,"%f\t",img_array[ro][cl]);
			}
		    fprintf(fp1,"\n");
		}


}
#else
#endif
