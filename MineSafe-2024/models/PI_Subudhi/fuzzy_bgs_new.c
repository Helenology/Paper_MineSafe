#include<stdio.h>
#include<malloc.h>
#include<math.h>
#include<time.h>
#include "read.h"
#define M 230
#define N 288
#define C 3
#define K  50
#define E  0.001
#define W 3
#define eph 0.003
#define fuz 100
#define sig 2
#define thr 1500
clock_t t1,t2;
#define med_window 1
#define mean_window 3

//#define NO_OF_IM 540
//#define NO_OF_DETECTION  675

#define NO_OF_IM 10
#define NO_OF_DETECTION  34
#define weight 0.5
#define alpha 0.3



				//Declaration of variables

double **a,**old_membership,**new_membership;
float ****temp1,***temp2;
FILE *fp1;			


			

int ***im2,***im3,***im6;
float ****mean,***im1,**im,***occur,****mean_temp,***dis,***dis_temp,***mu,***mu_temp;
double **cost;
int i,j,k,l,m,n,p,r,s,T=0,row,col,tp;
int level;
int uu,tt,aa,bb;
int iter;
char *str,*str1;     
float *threshold;
int **flag;
float temp5[50];
float ***im4,***im5;
				//function defined

void mean_filter();
void median_filter();



main()
{
double ti,tf;
t1=clock();
int w=((W-1)/2);
//printf("%d",w);
printf("hi hi hi\n");
FILE *fs0,*fs1,*fs2,*fs3,*fs4,*ft1,*ft2,*ft3,*ft4;
int a[C],b[C],c[C],valey_pos[C],peak1[C],peak2[C];
float **ar;
 ti=time(NULL);

int imx;	
FILE *fp_object,*fp_out;
FILE *fp_new;
		//array initialization
im=(float**)calloc(M,sizeof(float*));
im1=(float***)calloc(M,sizeof(float**));
im2=(int***)calloc(M,sizeof(int**));
im3=(int***)calloc(M,sizeof(int**));
im4=(float***)calloc(M,sizeof(float**));
im5=(float***)calloc(M,sizeof(float**));
im6=(int***)calloc(M,sizeof(int**));
ar=(float**)calloc(C,sizeof(float*));
flag=(int**)calloc(M,sizeof(int*));
occur=(float***)calloc(M,sizeof(float**));
mean=(float****)calloc(M,sizeof(float***));
mean_temp=(float****)calloc(M,sizeof(float***));
cost=(double**)calloc(M,sizeof(double*));
temp1=(float****)calloc(M,sizeof(float***));
temp2=(float***)calloc(M,sizeof(float**));
dis=(float***)calloc(M,sizeof(float**));
dis_temp=(float***)calloc(M,sizeof(float**));
mu=(float***)calloc(M,sizeof(float**));
mu_temp=(float***)calloc(M,sizeof(float**));
//printf("hi hi hi22222222222222\n");
	for(i=0;i<M;i++) {
		im[i]=(float*)calloc(N,sizeof(float));
		im1[i]=(float**)calloc(N,sizeof(float*));
		im2[i]=(int**)calloc(N,sizeof(int*));
		im3[i]=(int**)calloc(N,sizeof(int*));
		im4[i]=(float**)calloc(N,sizeof(float*));
		im5[i]=(float**)calloc(N,sizeof(float*));
		im6[i]=(int**)calloc(N,sizeof(int*));
		flag[i]=(int*)calloc(N,sizeof(int));
		occur[i]=(float**)calloc(N,sizeof(float*));
		mean[i]=(float***)calloc(N,sizeof(float**));
		mean_temp[i]=(float***)calloc(N,sizeof(float**));
		cost[i]=(double*)calloc(N,sizeof(double));
		temp1[i]=(float***)calloc(N,sizeof(float**));
		temp2[i]=(float**)calloc(N,sizeof(float*));
		dis[i]=(float**)calloc(N,sizeof(float*));
		dis_temp[i]=(float**)calloc(N,sizeof(float*));
		mu[i]=(float**)calloc(N,sizeof(float*));
		mu_temp[i]=(float**)calloc(N,sizeof(float*));
			}
		for(i=0;i<C;i++) {
		ar[i]=(float*)calloc((W*W),sizeof(float));}
//printf("hi hi hi3333333333333333\n");
	for(i=0;i<M;i++){
		for(j=0;j<N;j++){				
			im1[i][j]=(float*)calloc(C,sizeof(float));
			im2[i][j]=(int*)calloc(C,sizeof(int));
			im3[i][j]=(int*)calloc(C,sizeof(int));
			im4[i][j]=(float*)calloc(C,sizeof(float));
			im5[i][j]=(float*)calloc(C,sizeof(float));
			im6[i][j]=(int*)calloc(C,sizeof(int));
			occur[i][j]=(float*)calloc(K,sizeof(float));
			mean[i][j]=(float**)calloc(K,sizeof(float*));
			mean_temp[i][j]=(float**)calloc(K,sizeof(float*));
			temp1[i][j]=(float**)calloc(K,sizeof(float*));
			temp2[i][j]=(float*)calloc(C,sizeof(float));
			dis[i][j]=(float*)calloc(K,sizeof(float));
			dis_temp[i][j]=(float*)calloc(K,sizeof(float));
			mu[i][j]=(float*)calloc(K,sizeof(float));
			mu_temp[i][j]=(float*)calloc(K,sizeof(float));
		}}

	
	for(i=0;i<M;i++){
		for(j=0;j<N;j++){
                        for(k=0;k<K;k++){
			mean[i][j][k]=(float*)calloc(C,sizeof(float));
			mean_temp[i][j][k]=(float*)calloc(C,sizeof(float));
			temp1[i][j][k]=(float*)calloc(C,sizeof(float));
					}}}

printf("*****************Works On Initial Frame Starts Here*******************");


//fs0=fopen("WSur0440.ppm","r");
//fs0=fopen("over1785.ppm","r");

/*	for(i=0;i<M;i++){
		for(j=0;j<N;j++){
                        for(k=0;k<C;k++){
                                   fscanf(fs0,"%d\n",&im2[i][j][k]);
				        }
                                }
			}
*/
im2=img_read_head_ppm("pets6-3b0335.ppm");

mean_filter();

      for(k=0;k<C;k++)
         {
         for(p=0;p<(W*W);p++)
	    {
	    ar[k][p]=0;
	    }
         }
double temp4=0.0;
   for(i=w;i<M-w;i++)
      {
      for(j=w;j<N-w;j++)
         {
	 for(k=0;k<C;k++)
	    {
		 p=0;
		 for(m=i-w;m<i+w+1;m++)
		    {
		    for(n=j-w;n<j+w+1;n++)
			{
			ar[k][p]=im1[m][n][k];
			p=p+1;
			}
		    }
            }
      for(k=0;k<C;k++)
         {
         for(p=0;p<(W*W);p++)
	    {
	    mean[i][j][1][k]=mean[i][j][1][k]+ar[k][p];
	    }
         }
      for(k=0;k<C;k++)
         {
	 mean[i][j][1][k]=mean[i][j][1][k]/(W*W); 
         }
       for(k=0;k<C;k++)
         {
      	dis[i][j][1]=dis[i][j][1]+pow(fabsf(mean[i][j][1][k]-im1[i][j][k]),2.0);
	 }
        //temp4=cost[i][j];
        //mu[i][j][1]=1.0;
	//cost[i][j]=2*(1-exp(-dis[i][j][1]/pow(sig,2)));
	cost[i][j]=dis[i][j][1];
	//printf("%d\t%d\t%f\t%f\t%f\t%d\t%d\t%d\t%lf\t%lf\n",i,j,mean[i][j][0][0],mean[i][j][0][1],mean[i][j][0][2],im1[i][j][0],im1[i][j][1],im1[i][j][2],temp4,cost[i][j]);
	//printf("%d\t%d\t%lf\n",i,j,cost[i][j]);
	flag[i][j]=1;
	mu[i][j][1]=1.0;
        occur[i][j][1]=mu[i][j][1];
	}

    }
	

printf("\n***********Background Training starts here*********\n");
fp_new=fopen("bg_files_pets.txt", "r");
     for(iter=0;iter<NO_OF_IM;iter++)
        {
        str=(char*)calloc(256,sizeof(char));
        fscanf(fp_new,"%s",str);
        printf("taking input file:\n%s",str);
        im2=img_read_head_ppm(str);
	mean_filter();
	printf("**************************success********************************\n");
	float temp3=0;
	int temp4=0;
	float temp6=2.0/(float) (fuz-1);
	float temp7=0.0,temp8=0.0;
         
        for(i=w;i<M-w;i++)
            {
             for(j=w;j<N-w;j++)
                {
		 //temp5[flag[i][j]+1]=cost[i][j];
                 //for(T=1;T<(flag[i][j]+1);T++)
                   //  {

                      for(r=1;r<(flag[i][j]+1);r++)
		         {
                         for(k=0;k<C;k++)
                            {
			    mean_temp[i][j][r][k]=mean[i][j][r][k];
                            }
                         }
		      for(r=1;r<(flag[i][j]+1);r++)
                         {
                         dis_temp[i][j][r]=0.0;
                         }
		      for(r=1;r<(flag[i][j]+1);r++)
		         {
			 for(k=0;k<C;k++)
	 		    {
			    dis_temp[i][j][r]=dis_temp[i][j][r]+pow(fabsf(mean_temp[i][j][r][k]-im1[i][j][k]),2.0);
			    }
                          dis_temp[i][j][r]=sqrt(dis_temp[i][j][r]);
                         }     

		      for(r=1;r<(flag[i][j]+1);r++)
		         {
                         mu_temp[i][j][r]=0.0;
                         }
		      for(r=1;r<(flag[i][j]+1);r++)
		         {
                         mu_temp[i][j][r]=0.0;
			 if(dis_temp[i][j][r]==0.0)
                           {
                           for(s=1;s<(flag[i][j]+1);s++)
		             {
                              mu_temp[i][j][s]=0.0;
                             }
                           mu_temp[i][j][r]=1.0;
                           break;
                           }
                         else
                          {
                          temp3=0.0;
                          for(s=1;s<(flag[i][j]+1);s++)
		            {
                             temp3=temp3+pow((dis_temp[i][j][r]/dis_temp[i][j][s]),temp6);
			     //printf("%f\n",temp6);
			     //printf("%d\t%d\t%f\t%f\t%f\n",r,s,dis_temp[i][j][r],dis_temp[i][j][s],temp3);
                            }
			  mu_temp[i][j][r]=1/temp3;
                       // printf("%d\t%f\n",r,mu_temp[i][j][r]);
                          }
                        }
                      temp7=cost[i][j];
		      temp8=cost[i][j];
                      for(r=1;r<(flag[i][j]+1);r++)
		         {
		          temp7=temp7+pow(mu_temp[i][j][r],fuz)*pow(dis_temp[i][j][r],2);
                         }
                     // }


                        dis_temp[i][j][flag[i][j]+1]=0.0;
		 	for(k=0;k<C;k++)
			    {
			     p=0;
			     for(m=i-w;m<i+w+1;m++)
				{
				for(n=j-w;n<j+w+1;n++)
				   {
				   ar[k][p]=im1[m][n][k];
				   p=p+1;
				   }
			        }
			    }
			for(k=0;k<C;k++)
			   {
			   temp2[i][j][k]=0.0;
			   for(p=0;p<(W*W);p++)
			      {
			      temp2[i][j][k]=temp2[i][j][k]+ar[k][p];
			      }
			    }
		        for(k=0;k<C;k++)
			   {
			   mean_temp[i][j][flag[i][j]+1][k]=temp2[i][j][k]/(W*W);
			   }
                       for(r=1;r<(flag[i][j]+1);r++)
		         {
                         for(k=0;k<C;k++)
                            {
			    mean_temp[i][j][r][k]=mean[i][j][r][k];
                            }
                         }

		      for(r=1;r<(flag[i][j]+2);r++)
		         {
			 dis_temp[i][j][r]=0.0;
		             for(k=0;k<C;k++)
	 		        {
			        dis_temp[i][j][r]=dis_temp[i][j][r]+pow(fabsf(mean_temp[i][j][r][k]-im1[i][j][k]),2.0);
				}
                          dis_temp[i][j][r]=sqrt(dis_temp[i][j][r]);
                         }     

		      for(r=1;r<(flag[i][j]+2);r++)
		         {
                         mu_temp[i][j][r]=0.0;
                         }
		      for(r=1;r<(flag[i][j]+2);r++)
		         {
                         mu_temp[i][j][r]=0.0;
			 if(dis_temp[i][j][r]==0.0)
                           {
                           for(s=1;s<(flag[i][j]+2);s++)
		             {
                              mu_temp[i][j][s]=0.0;
                             }
                           mu_temp[i][j][r]=1.0;
                           break;
                           }
                         else
                          {
                          temp3=0.0;
                          for(s=1;s<(flag[i][j]+2);s++)
		            {
                             temp3=temp3+pow((dis_temp[i][j][r]/dis_temp[i][j][s]),temp6);
			     //printf("%f\n",temp6);
			     //printf("%d\t%d\t%f\t%f\t%f\n",r,s,dis_temp[i][j][r],dis_temp[i][j][s],temp3);
                            }
			  mu_temp[i][j][r]=1/temp3;
                        //printf("%d\t%f\n",r,mu_temp[i][j][r]);
                          }
                        }
                      for(r=1;r<(flag[i][j]+2);r++)
		         {
		         temp8=temp8+pow(mu_temp[i][j][r],fuz)*pow(dis_temp[i][j][r],2);
                         }
                      
		      //temp3=temp5[1];
		      //temp4=1;
		      //for(T=2;T<(flag[i][j]+2);T++)
			// {
    		         //if(fabsf(temp7-temp8)<thr)
			  //  {
			   // temp3=temp5[T];
			    //temp4=T;
			   // }	
    		        // }
		      //printf("%f\t%lf\n",temp7,temp8);
                      if((temp7<temp8) || (fabsf(temp7-temp8)<thr))
                        {
                        //printf("hi");
                        for(r=1;r<(flag[i][j]+1);r++)
		           {
                           for(k=0;k<C;k++)
	 		      {
        mean[i][j][r][k]=((occur[i][j][r]*mean[i][j][r][k])+(mu_temp[i][j][r]*im1[i][j][k]))/(occur[i][j][r]+mu_temp[i][j][r]);
                              }
                           occur[i][j][r]=occur[i][j][r]+mu_temp[i][j][r];
                           //
			   }
                        }
                      else
                       {
                        for(r=1;r<(flag[i][j]+1);r++)
		           {
                           for(k=0;k<C;k++)
	 		      {
        mean[i][j][r][k]=((occur[i][j][r]*mean[i][j][r][k])+(mu_temp[i][j][r]*im1[i][j][k]))/(occur[i][j][r]+mu_temp[i][j][r]);
                              }
                           occur[i][j][r]=occur[i][j][r]+mu_temp[i][j][r];
                           //
			   }
                      //printf("no");
                      for(k=0;k<C;k++)
	 		 {
                         mean[i][j][flag[i][j]+1][k]=mean_temp[i][j][flag[i][j]+1][k];
                         }  
                     occur[i][j][flag[i][j]+1]=mu_temp[i][j][flag[i][j]+1];                       
		     flag[i][j]++;
                     }
          
		}
	      }


		for(i=0;i<M;i++)
		      {
		      for(j=0;j<N;j++)
			 {
			 for(T=1;T<K;T++)
			    { 
			     temp5[T]=0.0;          
			     for(k=0;k<C;k++)
			      	 {
				 temp1[i][j][T][k]=0.0;
				 temp2[i][j][k]=0.0;
				 temp3=0;
				 temp4=0;
				 temp7=0.0;
				 temp8=0.0;

                                  }}}}

		    }

	
printf("\n***********Start of Background Subtraction starts here*********\n");
str1=(char*)calloc(256,sizeof(char));

     fp_object=fopen("detect_files_pets.txt", "r");
     fp_out=fopen("detect_files_pets_out.txt", "r");
	
	
     for(iter=0; iter<NO_OF_DETECTION;iter++)
        {
        fscanf(fp_object,"%s",str);
	fscanf(fp_out,"%s",str1);
printf(" %s hi going inside\n",str);
printf(" %s hi going inside\n",str1);
printf("\n*****starting object detection for file number :%d\n",iter); 	
	im2=img_read_head_ppm(str);
	mean_filter();
	printf("**************************success********************************\n");
	float temp3=0;
	int temp4=0;
	float temp6=0.0;
	float temp7=0.0,temp8=0.0;
         
        for(i=w;i<M-w;i++)
            {
             for(j=w;j<N-w;j++)
                {
		 //temp5[flag[i][j]+1]=cost[i][j];
                 //for(T=1;T<(flag[i][j]+1);T++)
                   //  {

                      for(r=1;r<(flag[i][j]+1);r++)
		         {
                         for(k=0;k<C;k++)
                            {
			    mean_temp[i][j][r][k]=mean[i][j][r][k];
                            }
                         }
		      for(r=1;r<(flag[i][j]+1);r++)
                         {
                         dis_temp[i][j][r]=0.0;
                         }
		      for(r=1;r<(flag[i][j]+1);r++)
		         {
			 for(k=0;k<C;k++)
	 		    {
			    dis_temp[i][j][r]=dis_temp[i][j][r]+pow(fabsf(mean_temp[i][j][r][k]-im1[i][j][k]),2.0);
			    }
                          dis_temp[i][j][r]=sqrt(dis_temp[i][j][r]);
                         }     

		      for(r=1;r<(flag[i][j]+1);r++)
		         {
                         mu_temp[i][j][r]=0.0;
                         }
		      for(r=1;r<(flag[i][j]+1);r++)
		         {
                         mu_temp[i][j][r]=0.0;
			 if(dis_temp[i][j][r]==0.0)
                           {
                           for(s=1;s<(flag[i][j]+1);s++)
		             {
                              mu_temp[i][j][s]=0.0;
                             }
                           mu_temp[i][j][r]=1.0;
                           break;
                           }
                         else
                          {
                          temp3=0.0;
                          for(s=1;s<(flag[i][j]+1);s++)
		            {
                             temp3=temp3+pow((dis_temp[i][j][r]/dis_temp[i][j][s]),temp6);
			     //printf("%f\n",temp6);
			     //printf("%d\t%d\t%f\t%f\t%f\n",r,s,dis_temp[i][j][r],dis_temp[i][j][s],temp3);
                            }
			  mu_temp[i][j][r]=1/temp3;
                       // printf("%d\t%f\n",r,mu_temp[i][j][r]);
                          }
                        }
                      temp7=cost[i][j];
		      temp8=cost[i][j];
                      for(r=1;r<(flag[i][j]+1);r++)
		         {
		          temp7=temp7+pow(mu_temp[i][j][r],fuz)*pow(dis_temp[i][j][r],2);
                         }
                     // }


                       // dis_temp[i][j][flag[i][j]+1]=0.0;
		 	for(k=0;k<C;k++)
			    {
			     p=0;
			     for(m=i-w;m<i+w+1;m++)
				{
				for(n=j-w;n<j+w+1;n++)
				   {
				   ar[k][p]=im1[m][n][k];
				   p=p+1;
				   }
			        }
			    }
			for(k=0;k<C;k++)
			   {
			   temp2[i][j][k]=0.0;
			   for(p=0;p<(W*W);p++)
			      {
			      temp2[i][j][k]=temp2[i][j][k]+ar[k][p];
			      }
			    }
		        for(k=0;k<C;k++)
			   {
			   mean_temp[i][j][flag[i][j]+1][k]=temp2[i][j][k]/(W*W);
			   }
                       for(r=1;r<(flag[i][j]+1);r++)
		         {
                         for(k=0;k<C;k++)
                            {
			    mean_temp[i][j][r][k]=mean[i][j][r][k];
                            }
                         }

		      for(r=1;r<(flag[i][j]+2);r++)
		         {
			 dis_temp[i][j][r]=0.0;
		             for(k=0;k<C;k++)
	 		        {
			        dis_temp[i][j][r]=dis_temp[i][j][r]+pow(fabsf(mean_temp[i][j][r][k]-im1[i][j][k]),2.0);
				}
                          dis_temp[i][j][r]=sqrt(dis_temp[i][j][r]);
                         }     

		      for(r=1;r<(flag[i][j]+2);r++)
		         {
                         mu_temp[i][j][r]=0.0;
                         }
		      for(r=1;r<(flag[i][j]+2);r++)
		         {
                         mu_temp[i][j][r]=0.0;
			 if(dis_temp[i][j][r]==0.0)
                           {
                           for(s=1;s<(flag[i][j]+2);s++)
		             {
                              mu_temp[i][j][s]=0.0;
                             }
                           mu_temp[i][j][r]=1.0;
                           break;
                           }
                         else
                          {
                          temp3=0.0;
                          for(s=1;s<(flag[i][j]+2);s++)
		            {
                             temp3=temp3+pow((dis_temp[i][j][r]/dis_temp[i][j][s]),temp6);
			     //printf("%f\n",temp6);
			     //printf("%d\t%d\t%f\t%f\t%f\n",r,s,dis_temp[i][j][r],dis_temp[i][j][s],temp3);
                            }
			  mu_temp[i][j][r]=1/temp3;
                       // printf("%d\t%f\n",r,mu_temp[i][j][r]);
                          }
                        }
                      for(r=1;r<(flag[i][j]+2);r++)
		         {
		         temp8=temp8+pow(mu_temp[i][j][r],fuz)*pow(dis_temp[i][j][r],2);
                         }
                      

		      //temp3=temp5[1];
		      //temp4=1;
		      //for(T=2;T<(flag[i][j]+2);T++)
			// {
    		         //if(fabsf(temp7-temp8)<thr)
			  //  {
			   // temp3=temp5[T];
			    //temp4=T;
			   // }	
    		        // }
		      //printf("%d\t%lf\n",temp4,temp5[temp4]);
                     /* if((temp7<temp8)&& (fabsf(temp7-temp8)>thr))
                        {
                           for(k=0;k<C;k++)
	 		      {
                              im3[i][j][k]=255;
                              }
                          //printf("hi");
                         for(r=1;r<(flag[i][j]+1);r++)
		           {
                           for(k=0;k<C;k++)
	 		      {
        mean[i][j][r][k]=((occur[i][j][r]*mean_temp[i][j][r][k])+(mu_temp[i][j][r]*im1[i][j][k]))/(occur[i][j][r]+mu_temp[i][j][r]);
                              }
                           occur[i][j][r]=occur[i][j][r]+mu_temp[i][j][r];
			   }
                        }
                      else{
                          for(k=0;k<C;k++)
	 		     {
                             im3[i][j][k]=255;
                             }
                          }*/

 			if((temp7<temp8) || (fabsf(temp7-temp8)<thr))
                        {
                           for(k=0;k<C;k++)
	 		      {
                              im3[i][j][k]=0;
                              }
                         }                        
                      else{
                          for(k=0;k<C;k++)
	 		     {
                             im3[i][j][k]=255;
                             }
                          }


		}
	      }

median_filter();


//img_write_ppm(im3,str1);
img_write_ppm(im6,str1);

for(i=0;i<M;i++){

	for(j=0;j<N;j++){
                for(k=0;k<C;k++){
                           im3[i][j][k]=0;
			   im1[i][j][k]=0;
			        }

                        }
                }





for(i=0;i<M;i++)
      {
      for(j=0;j<N;j++)
         {
         for(T=1;T<K;T++)
            { 
             temp5[T]=0.0;          
	     for(k=0;k<C;k++)
	      	 {
		 mean_temp[i][j][T][k]=0.0;
                 temp1[i][j][T][k]=0.0;
                 temp2[i][j][k]=0.0;
                 temp3=0;
                 temp4=0;}}}}
printf("**************************success********************************\n");	

} 
}

void mean_filter()
{
int ro,cl,dp;
int xo,yo,zo,uu;

   if(mean_window==1)
	{
        for(ro=0;ro<M;ro++)
          {
          for(cl=0;cl<N;cl++)
            {
            for(dp=0;dp<C;dp++)
               {
                im1[ro][cl][dp]=im2[ro][cl][dp];
               }
            }
           
          }
        }
else
{
int aa=(mean_window-1)/2;
   for(ro=0;ro<M;ro++)
      {
      for(cl=0;cl<N;cl++)
         {
         for(dp=0;dp<C;dp++)
            {
             im1[ro][cl][dp]=0.0;}}}
   for(ro=aa;ro<M-aa;ro++)
      {
      for(cl=aa;cl<N-aa;cl++)
         {
         for(dp=0;dp<C;dp++)
            {
	    for(xo=ro-aa;xo<ro+aa+1;xo++)
	       {
	       for(yo=cl-aa;yo<cl+aa+1;yo++)
		  {
		  im1[ro][cl][dp]=im1[ro][cl][dp]+(float) im2[xo][yo][dp];
		  }
	       }
            im1[ro][cl][dp]=im1[ro][cl][dp]/(mean_window*mean_window);
	    //printf("%f\n",im1[ro][cl][dp]);
            }
         }
       }
   }       

  }




void median_filter()
{
int *ar;
int ro,cl,dp,m;



if(med_window==1)
	{
        for(ro=0;ro<M;ro++)
          {
          for(cl=0;cl<N;cl++)
            {
            for(dp=0;dp<C;dp++)
               {
                im6[ro][cl][dp]=im3[ro][cl][dp];
               }
            }
           
          }
        }

else
{
aa=(med_window-1)/2;
bb=(((med_window*med_window)-1)/2)+1;
ar=(int*)calloc((med_window*med_window),sizeof(int));
for(dp=0;dp<C;dp++)
   {
   for(ro=aa;ro<M-aa;ro++)
      {
      for(cl=aa;cl<N-aa;cl++)
         {
	 m=0;
	 for(i=ro-aa;i<ro+aa+1;i++)
	    {
	    for(j=cl-aa;j<cl+aa+1;j++)
		{
		ar[m]=im3[i][j][dp];
		m=m+1;
		}
	    }
	 for(i=0;i<(med_window*med_window);i++)
	    {
	    for(j=0;j<(med_window*med_window)-1;j++)
	       {
	       if(ar[j]>=ar[j+1])
		 {
		  uu=ar[j+1];
		  ar[j+1]=ar[j];
		  ar[j]=uu;
                 }
               }
            }
         im6[ro][cl][dp]=ar[bb];
          }
       }
    }
}

  }





